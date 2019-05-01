import cv2
import numpy as np


def hsv(frame):

    colour_space = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    return colour_space


def gray(frame):

    colour_space = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return colour_space


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result
