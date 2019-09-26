import cv2
import numpy as np
import alglib.colour_space as colour
from scipy.signal import find_peaks

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


def contour_curvature_kcos(contour, vectSize, stepSize):

    kcospoints = []

    points_in_contour = len(contour)
    
    steps = int(np.floor(points_in_contour / stepSize))
    kcospoints.append(0)

    if steps <= 2:
        return kcospoints

    for i in range(vectSize, steps - 1*vectSize):

        XNminus = contour[i*stepSize - 1*vectSize]
        X = contour[i]
        XNplus = contour[i*stepSize + 1*vectSize]

        A = [(X[0][0] - XNminus[0][0]), (X[0][1] - XNminus[0][1])]
        B = [-(XNplus[0][0] - X[0][0]), -(XNplus[0][1] - X[0][1])]

        A_B_Dot = np.dot(A, B)

        A_abs = np.sqrt(A[0]*A[0] + A[1]*A[1])
        B_abs = np.sqrt(B[0]*B[0] + B[1]*B[1])

        vect_abs = A_abs * B_abs

        kcos = A_B_Dot/vect_abs

        kcospoints.append(1 - kcos)

    kcospoints.append(0)

    cospoints = []

    for point in kcospoints:
        cospoints.append(np.arccos(point))

    return kcospoints


def contour_angle_maxima(contour, vectSize = 15, stepSize = 1):

    angular_deriv = contour_curvature_kcos(contour, vectSize, stepSize)

    if len(angular_deriv) > 1:
        maxima, _ = find_peaks(angular_deriv, height=max(angular_deriv)*0.1)

    else:
        maxima = []

    contour_points = []

    for max_point in maxima:
        contour_points.append([contour[max_point*stepSize]])

    return maxima, angular_deriv, contour_points


def contour_clustering(contour, cluster_points, grouping_distance=15):

    contour_list = contour.tolist()

    cluster_points_list = cluster_points[0].tolist()

    contour_indexs = []

    for point in cluster_points_list:

        index = contour_list.index(point)

        contour_indexs.append(index)

    contour_indexs = sorted(contour_indexs)

    cluster_groups = []

    current_group = []

    if len(contour_indexs) - 1 > 1:

        for i in range(0, len(contour_indexs)):

            found_neighbour = False

            if i == 0:
                point_to_add = contour_list[contour_indexs[i]]
                current_group.append(point_to_add)

            A_dist = contour_indexs[i - 1]
            B_dist = contour_indexs[i]

            point_to_point_dist = abs(A_dist - B_dist)

            if point_to_point_dist < grouping_distance:
                found_neighbour = True
                point_to_add = contour_list[contour_indexs[i]]
                current_group.append(point_to_add)

            if found_neighbour == False:
                cluster_groups.append(current_group)
                current_group = []
                current_group.append(contour_list[contour_indexs[i]])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    for i in range(0, len(cluster_groups)):

        if len(cluster_groups[i]) > 1:
            np_group = np.array(cluster_groups[i], dtype=int)
            grouped_point = centeroidnp(np_group)
            X = grouped_point[0]
            Y = grouped_point[1]
            cluster_groups[i] = [[[X, Y]]]

    return cluster_groups


def centeroidnp(arr):
    points = arr.tolist()

    x_sum = 0
    y_sum = 0

    length = len(points)

    for point in points:
        x_sum += point[0][0]
        y_sum += point[0][1]

    return int(x_sum/length), int(y_sum/length)


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

    dark = hsv[..., 2] < 32
    hsv[dark] = 0

    h = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    h = np.clip(h * 0.005 * hist_scale, 0, 1)

    return hsv_map * h[:, :, np.newaxis] / 255.0
