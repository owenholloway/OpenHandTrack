import cv2
import numpy as np


def hsv_mask(frame):

    colour_space = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([2, 35, 128], np.uint8)
    upper = np.array([30, 124, 255], np.uint8)

    mask = cv2.inRange(colour_space, lower, upper)
    #mask = cv2.erode(mask, None, iterations=3)
    #mask = cv2.dilate(mask, None, iterations=3)

    return mask