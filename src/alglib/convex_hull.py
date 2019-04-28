import cv2
import numpy as np


def get_hull_points(contour):

    hull = cv2.convexHull(contour, returnPoints=False)
    hullPoints = cv2.convexHull(contour, returnPoints=True)

    try:
        defects = cv2.convexityDefects(contour, hull)
    except Exception:
        defects = None

    return hullPoints, hull, defects
