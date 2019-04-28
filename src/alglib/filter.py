import cv2
import numpy as np


def guass(frame):

    kernal = cv2.getGaussianKernel(15, 1)

    frame = cv2.filter2D(frame, -1, kernal)

    return frame


def sharpen(frame):

    kernal = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    frame = cv2.filter2D(frame, -1, kernal)

    return frame
