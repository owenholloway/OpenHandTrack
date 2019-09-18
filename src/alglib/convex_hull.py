import cv2
import numpy as np


def get_hull_points(contour):

    hull = cv2.convexHull(contour, returnPoints=False)
    hull_points = cv2.convexHull(contour, returnPoints=True)

    try:
        defects = cv2.convexityDefects(contour, hull)
    except Exception:
        defects = None

    return hull_points, hull, defects


#def cluster_points(hull_points):
#
#    thresh = 30
#    clustered_points = []
#
#    uncomputed_points = list.clear(hull_points)
#
#    while (len(uncomputed_points) > 0):
#
#        point_of_interest = uncomputed_points[0]
#
#        neibours = [];
#
#        list.remove(point_of_interest)
#
#        for point in uncomputed_points:
#
#            if (np.linalg.norm(point_of_interest-point) < thresh):
#
#
#
#        clustered_points.append(neibours)
#
#
#    return clustered_points


def draw_hull_on_frame(frame, contour):

    hull_points, hull, defect_points = get_hull_points(contour)

    hull_point_no = len(hull_points)
    if hull_point_no > 0:

        for i in range(0, hull_point_no):

            cv2.circle(frame, (int(hull_points[i][0][0]), int(hull_points[i][0][1])), int(10), (0, 255, 255), 2)

            if i == hull_point_no - 1:
                cv2.line(frame, (hull_points[i][0][0], hull_points[i][0][1]),
                         (hull_points[0][0][0], hull_points[0][0][1]), (255, 255, 255), 1)

            else:
                cv2.line(frame, (hull_points[i][0][0], hull_points[i][0][1]),
                         (hull_points[i + 1][0][0], hull_points[i + 1][0][1]), (255, 255, 255), 1)
