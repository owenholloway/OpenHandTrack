import cv2
import numpy as np


def get_hull_points(contour):

    hull = cv2.convexHull(contour, False)

    cnt = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    try:
        defects = cv2.convexityDefects(contour, hull)
    except Exception:
        defects = None

    return hull, defects
