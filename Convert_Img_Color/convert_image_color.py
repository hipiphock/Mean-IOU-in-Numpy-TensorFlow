import os
import sys
import time
from config import RL_Obj_List
import numpy as np
import cv2
import logging
IMAGE_CHANNEL = 3

logger = logging.getLogger("seg_detector")

def convert_color_seg(grey_label_array):
    t1 = time.time()

    height = grey_label_array.shape[0]
    width = grey_label_array.shape[1]
    channel = IMAGE_CHANNEL

    shape = (height, width, channel)
    color_label = np.zeros(shape=shape, dtype=np.uint8)

    for i in np.unique(grey_label_array):
        list0 = np.where(grey_label_array == i)
        color = RL_Obj_List[i][1]
        color_label[list0[0], list0[1]] = color

    t2 = time.time()

    logger.info("time spend : {:07.4f} on convert_color_seg".format(t2-t1))
    cv2.imshow("color_seg_show", color_label)
    cv2.waitKey(1)
    return np.copy(color_label)

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # read images
    for idx, path in enumerate(os.listdir("target/")):
        image = cv2.imread("target/" + path, -1)
        print("working on target/" + path)
        output_image = convert_color_seg(image)
        cv2.imwrite("color_target/" + path, output_image)

        image = cv2.imread("test/" + path, -1)
        print("working on test/" + path)
        output_image = convert_color_seg(image)
        cv2.imwrite("color_test/" + path, output_image)