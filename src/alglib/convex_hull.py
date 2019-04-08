import cv2
import numpy as np


def getHullPoints(contour):

    hull = cv2.convexHull(contour, False)

    return hull
