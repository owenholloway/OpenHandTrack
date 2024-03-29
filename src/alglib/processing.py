import cv2
import numpy as np
import alglib.colour_space as colour


def hsv_mask(frame, lower=np.array([2, 35, 128], np.uint8), upper=np.array([30, 124, 255], np.uint8)):

    colour_space = colour.hsv(frame)

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


def contour_hsv(frame, lower=np.array([2, 35, 128], np.uint8), upper=np.array([30, 124, 255], np.uint8)):

    mask = hsv_mask(frame, lower, upper)

    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


def contour_canny(frame, delta=0.33):

    canny = auto_canny(frame, delta)

    return cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]


def blob_detect(frame):

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create()

    return detector.detect(frame)


def filter_contours(contours, area=50):

    rtn_contours = []

    for cnt in contours:
        if area < cv2.contourArea(cnt):
            rtn_contours.append(cnt)

    return rtn_contours


def hand_contours():

    return


hsv_map = np.zeros((180, 256, 3), np.uint8)
h, s = np.indices(hsv_map.shape[:2])
hsv_map[:, :, 0] = h
hsv_map[:, :, 1] = s
hsv_map[:, :, 2] = 255
hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)
hist_scale = 10


def hsv_histogram(hsv):

    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = hsv[..., 2] < 32
    hsv[dark] = 0

    h = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    h = np.clip(h * 0.005 * hist_scale, 0, 1)

    return hsv_map * h[:, :, np.newaxis] / 255.0
