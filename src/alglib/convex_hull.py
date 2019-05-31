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


def draw_hull_on_frame(frame, contour):

    points, hull, defects = get_hull_points(contour)

    hull_points = len(points)
    if hull_points > 0:

        for i in range(0, hull_points):

            cv2.circle(frame, (int(points[i][0][0]), int(points[i][0][1])), int(10), (0, 255, 255), 2)

            if i == hull_points - 1:
                cv2.line(frame, (points[i][0][0], points[i][0][1]),
                         (points[0][0][0], points[0][0][1]), (255, 255, 255), 1)

            else:
                cv2.line(frame, (points[i][0][0], points[i][0][1]),
                         (points[i + 1][0][0], points[i + 1][0][1]), (255, 255, 255), 1)
