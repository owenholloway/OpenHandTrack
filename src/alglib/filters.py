import cv2
import numpy as np


def hsv_mask(frame):

    colour_space = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([2, 35, 128], np.uint8)
    upper = np.array([30, 124, 255], np.uint8)

    mask = cv2.inRange(colour_space, lower, upper)

    return mask, colour_space


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged
