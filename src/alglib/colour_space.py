import cv2


def hsv(frame):

    colour_space = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    return colour_space


def gray(frame):

    colour_space = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return colour_space
