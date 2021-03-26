import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import cv2
import numpy as np

# set up paths
CWD = os.getcwd()
MODEL_PATH = os.path.join(CWD, 'model3')
CHECKPOINT_PATH = os.path.join(MODEL_PATH, 'checkpoints')
CONFIG_PATH = os.path.join(MODEL_PATH, 'config\pipeline.config')
ANNOTATION_PATH = os.path.join(MODEL_PATH, 'annotations')
VIDEO_PATH = os.path.join(CWD, 'video')
OUTPUT_VIDEO_PATH = os.path.join(MODEL_PATH, 'motion_detection_videos')

if not os.path.isdir(OUTPUT_VIDEO_PATH):
    os.mkdir(OUTPUT_VIDEO_PATH)

# constants for detecting waves
FRAME_THRESHOLD = 20 # 1 seconds
WAVE_FRAME_PATIENCE = 5
CENTROID_THRESHOLD = 50 # I will tweak this number as I experiment

# ---------------------------- define all of the helper functions ------------------------
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def get_centroid_from_dims(ymin, xmin, ymax, xmax, size):
    """
    The coordinates that are inputted are normalized, so we multiply by the img width and height.
    Then, the centroid is in the center of the rectangle, which is the average of the x's and y's
    """
    width = size[0]
    height = size[1]
    left, right, top, bottom = xmin*width, xmax*width, ymin*height, ymax*height
    return ((left + right)/2.0, (top + bottom)/2.0)

def average_wave_length(wave_length, lefts, rights, is_left = True):
    if is_left:
        lefts.append(wave_length)
    else:
        rights.append(wave_length)

def determine_direction(initial_position, last_position, overall_wave_count, wave_length, lefts, rights, size = (1280, 720)):
    """
    We can easily determine whether the detected wave is a left or right by the final and initial position of the x values
    """
    ymin, xmin, ymax, xmax = initial_position
    initial_centroid = get_centroid_from_dims(ymin, xmin, ymax, xmax, size)
    ymin, xmin, ymax, xmax = last_position
    final_centroid = get_centroid_from_dims(ymin, xmin, ymax, xmax, size)

    x0 = initial_centroid[0]
    x1 = final_centroid[0]

    if x1 - x0 < 0:
        overall_wave_count[1] += 1 # then it's a right
        average_wave_length(wave_length, lefts, rights, is_left = False)
    else:
        overall_wave_count[0] += 1 # then it's a left
        average_wave_length(wave_length, lefts, rights)
    return overall_wave_count

def keep_longest_wave(ongoing_detections):
    """
    If a left and a right intersect, the centroid algorithm will call them two seperate waves and count the same wave twice.
    To avoid this, we will filter and keep only the longest wave
    """
    passed = []
    indices_to_keep = []
    for ind, sublist in enumerate(ongoing_detections):
        if len(sublist) == 1:
            passed.append(True)
        else:
            list_of_frames = []
            passed.append(False)
            first_frame = sublist[0][-1]
            final_box = sublist[-1]
            final_frame = final_box[-1]
            list_of_frames.append(first_frame)
            for ind2 in range(ind + 1, len(ongoing_detections)):
                if ind2 == len(ongoing_detections) - 1:
                    pass
                else:
                    if final_box in ongoing_detections[ind2]:
                        initial_frame = ongoing_detections[ind2][0][-1]
                        list_of_frames.append(initial_frame)
                index_to_keep = list_of_frames.index(min(list_of_frames))
                indices_to_keep.append(index_to_keep)
    if sum(passed) == len(passed):
        return ongoing_detections
    ongoing_detections = [sublist for ind, sublist in enumerate(ongoing_detections) if ind in indices_to_keep]
    return ongoing_detections

def update_wave_count(current_frame, ongoing_detections, overall_wave_count, box_dict, owv, lefts, rights):
    """
    I declare a detection a wave if: 
    1) at least 5 frames (0.25 seconds) pass since the detection ended
    2) the wave lasts for more than 20 frames (1 second)

    We pass the detections into the keep_longest_wave function to ensure we are not counting a wave more than once.
    """
    if len(ongoing_detections) > 1:
        ongoing_detections = keep_longest_wave(ongoing_detections)
    for ind, sublist in enumerate(ongoing_detections):
        last_frame = sublist[-1][1]
        last_position = sublist[-1][0]
        if current_frame - last_frame > WAVE_FRAME_PATIENCE:
            # check to see that overall the wave lasted at least FRAME_THRESHOLD
            first_frame = sublist[0][1]
            initial_position = sublist[0][0]
            if last_frame - first_frame > FRAME_THRESHOLD:
                owv += 1
                wave_length = (last_frame - first_frame)/20 # 20 FPS
                overall_wave_count = determine_direction(initial_position, last_position, overall_wave_count, wave_length, lefts, rights)
                # remove the sublist from ongoing detections
                ongoing_detections.remove(sublist)
            else:
                ongoing_detections.remove(sublist)
                if ongoing_detections is None:
                    ongoing_detections = list()
    return ongoing_detections, overall_wave_count, owv, lefts, rights

def add_to_ongoing_detections(current_box, previous_box, ongoing_detections, current_frame, previous_frame):
    """
    If a centroid is found, we pair the current detection box with a previous detection box
    """
    if ongoing_detections is not None:
        for sublist in ongoing_detections:
            for pair in sublist:
                if previous_box in pair:
                    if len(sublist) == 1:
                        sublist.append([current_box, current_frame])
                    else:
                        sublist[-1] = [current_box, current_frame]
    return ongoing_detections               


def find_closest_centroid(current_box, previous_boxes, ongoing_detections, current_frame, previous_frame, size):
    """
    "Closest" centroid is found by calculating the euclidean distance between all detections from two frames.
    The centroids with the minimum distance will be paired if the distance is less than CENTROID_THRESHOLD. This
    ensures that a wave is not paired with a different wave that is relatively close.
    """
    ymin0, xmin0, ymax0, xmax0 = current_box
    current_centroid = np.asarray(get_centroid_from_dims(ymin0, xmin0, ymax0, xmax0, size))
    current_min = np.inf
    min_index = 0
    for ind, previous_box in enumerate(previous_boxes): # calculate the distance between all previous detection boxes
        ymin1, xmin1, ymax1, xmax1 = previous_box
        previous_centroid = np.asarray(get_centroid_from_dims(ymin1, xmin1, ymax1, xmax1, size))
        dist = np.linalg.norm(current_centroid - previous_centroid)
        if dist < current_min:
            current_min = dist # keep the closes centroid
            min_index = ind
    if current_min < CENTROID_THRESHOLD: # then we found a box closest to the previous frame
        closest_box = previous_boxes[min_index]
        ongoing_detections = add_to_ongoing_detections(current_box, closest_box, ongoing_detections, current_frame, previous_frame)

    else:  # then this is a new wave, so append to the ongoing detections
        ongoing_detections.append([[current_box, current_frame]])
    return ongoing_detections

# create function to detect whether the detection boxes intersect through frames
def loop_detected_frames(current_frame, box_dict, ongoing_detections, size):
    """
    We check it the current frame's detection boxes have a close centroid in the previous frames.
    We give a buffer of 5 frames just in case the model does not detect a wave from one frame to another.
    """
    previous_frames = range(current_frame - 1, current_frame - WAVE_FRAME_PATIENCE - 1, -1) # check to see if the box intersects with any box up to 5 frames ago
    previous_frame_in_keys = False # we loop through all previous frames, but only need one to be true

    for previous_frame in previous_frames:
        # double for loop
        if previous_frame in box_dict.keys(): # this ensures we have a corresponding key in the dict
            previous_frame_in_keys = True
            for current_box in box_dict[current_frame]: # each frame may have more than one wave, so we loop through
                ongoing_detections = find_closest_centroid(current_box, box_dict[previous_frame], ongoing_detections, current_frame, previous_frame, size)
    if not previous_frame_in_keys: # then the wave may be a new one, so we add to ongoing detections and deal with it later
        for current_box in box_dict[current_frame]:
            ongoing_detections.append([[current_box, current_frame]])
    return ongoing_detections


# -------------------------------------- end of helper functions -----------------------------------------------

# load the trained model from checkpoint
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config = configs['model'], is_training = False)

ckpt = tf.compat.v2.train.Checkpoint(model = detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')

video_name = 'hb_vid_exciting.mp4'
video = os.path.join(VIDEO_PATH, video_name)

cap = cv2.VideoCapture(video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

# output video
result = cv2.VideoWriter(os.path.join(OUTPUT_VIDEO_PATH, 'centoid.mp4'),  
                         cv2.VideoWriter_fourcc(*'XVID'), 
                         20.0, size) 

# initiate constants before starting analysis
n = 0
ongoing_detections = list()
box_dict = dict()
overall_wave_count = [0,0] # left, right
owv = 0
lefts = []
rights = []
while True: 
    ret, frame = cap.read()
    if ret:
        image_np = np.array(frame)
    
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.5,
                    agnostic_mode=False)

        boxes = detections['detection_boxes'] # get the detection boxes and scores
        scores = detections['detection_scores']
        boxes_filtered = [box for box, score in zip(boxes, scores) if score > 0.5] # keep only the boxes whose score is greater than 0.5
        # prepare to display text on image
        left_count_text = 'Lefts: '
        right_count_text = 'Rights: '
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        left_org = (75, 50)
        left_avg_org = (10, 100)
        right_org = (600,50)
        right_avg_org = (530, 100)
        # fontScale
        fontScale = 1
        fontScaleAvg = 0.75
        # BGR
        left_color = (0, 255, 0) # green
        right_color = (0, 255, 255) # yellow
        # Line thickness of 2 px
        thickness = 2
    
        # define difference of image
        delta = cv2.absdiff(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY))

        if delta.sum() != 0: # this means there was a detection
            list_of_coordinates = []
            for box in boxes_filtered:
                ymin = box[0]
                xmin = box[1]
                ymax = box[2]
                xmax = box[3]
                box_dims = [ymin, xmin, ymax, xmax]
                list_of_coordinates.append(box_dims)
            box_dict[n] = list_of_coordinates
            ongoing_detections = loop_detected_frames(n, box_dict, ongoing_detections, size)

        if len(ongoing_detections) != 0:
            # find out if the wave ended or is still going
            ongoing_detections, overall_wave_count, owv, lefts, rights = update_wave_count(n, ongoing_detections, overall_wave_count, box_dict, owv, lefts, rights)

        # Using cv2.putText() method
        # update left counter
        image = cv2.putText(cv2.resize(image_np_with_detections, (800, 600)), 
                            left_count_text + '{}'.format(overall_wave_count[0]), left_org, font, 
                            fontScale, left_color, thickness, cv2.LINE_AA)

        # average_wave_length for lefts
        if len(lefts) == 0:
            avg_lefts = 0
        else:
            avg_lefts = sum(lefts)/len(lefts)
        image = cv2.putText(image, 
                            'AVG Wave Time: {:.1f} s'.format(avg_lefts), left_avg_org, font, 
                            fontScaleAvg, left_color, thickness, cv2.LINE_AA)

        # update right counter
        image = cv2.putText(image, 
                            right_count_text + '{}'.format(overall_wave_count[1]), right_org, font, 
                            fontScale, right_color, thickness, cv2.LINE_AA)

        # average wave length for rights
        if len(rights) == 0:
            avg_rights = 0
        else:
            avg_rights = sum(rights)/len(rights)

        image = cv2.putText(image, 
                            'AVG Wave Time: {:.1f} s'.format(avg_rights), right_avg_org, font, 
                            fontScaleAvg, right_color, thickness, cv2.LINE_AA)
        cv2.imshow('Wave Motion Detection', image)
        # cv2.imshow('delta', cv2.resize(delta, (800,600)))
        result.write(cv2.resize(image, size))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    n += 1
result.release()
cap.release()