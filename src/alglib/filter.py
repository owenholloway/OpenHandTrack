import cv2
import numpy as np


def guass(frame, sigma=1):

    kernel = cv2.getGaussianKernel(5, sigma)

    frame = cv2.filter2D(frame, -1, kernel)

    return frame


def sharpen(frame):

    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    frame = cv2.filter2D(frame, -1, kernel)

    return frame
