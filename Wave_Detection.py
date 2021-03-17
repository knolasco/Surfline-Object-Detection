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

# load the trained model from checkpoint
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config = configs['model'], is_training = False)

ckpt = tf.compat.v2.train.Checkpoint(model = detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')

video_name = 'hb_exciting3.mp4'
video = os.path.join(VIDEO_PATH, video_name)

cap = cv2.VideoCapture(video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

result = cv2.VideoWriter(os.path.join(MODEL_PATH, 'res2.mp4'),  
                         cv2.VideoWriter_fourcc(*'XVID'), 
                         20.0, size) 
while True: 
    ret, frame = cap.read()
    if ret:
        image_np = np.array(frame)

        # this is where I would process the image_np (increase contrast and make black and white)
        # then we plut that into tf.convert_to_tensor
    
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

        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (1000, 750)))
        result.write(cv2.resize(image_np_with_detections, size))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


result.release()
cap.release()
