import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

CWD = os.getcwd()
TRAIN_PATH = CWD + r'\train3\\'
TEST_PATH = CWD + r'\test_2\\'
SAVE_TRAIN_PATH = CWD +  r'\flipped_train\\'
SAVE_TEST_PATH = CWD +  r'\flipped_test\\'


def flip_images(path, path_to_save):
    for img in os.listdir(path):
        if '.JPG' in img:
            tmp_img = mpimg.imread(path + img)
            img_flipped = np.flip(tmp_img, 1)
            img_flipped = Image.fromarray(img_flipped)
            img_flipped.save(path_to_save + img.replace('.JPG','') + '_flip.JPG')

for img_path, img_save_path in zip([TRAIN_PATH, TEST_PATH], [SAVE_TRAIN_PATH, SAVE_TEST_PATH]):
    flip_images(img_path, img_save_path)