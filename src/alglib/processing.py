import cv2
import numpy as np
import alglib.colour_space as colour


def hsv_mask(frame):

    colour_space = colour.hsv(frame)

    lower = np.array([2, 35, 128], np.uint8)
    upper = np.array([30, 124, 255], np.uint8)

    mask = cv2.inRange(colour_space, lower, upper)

    return mask


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def contour_hsv(frame):

    mask = hsv_mask(frame)

    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


def contour_canny(frame):

    canny = auto_canny(frame)

    return cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
