# University of Tasmania
# Owen Holloway
# 206891
# 2019
import cv2
import numpy as np
import alglib.processing as processing
import alglib.two_filter as two_filter
import alglib.colour_space as colour
import alglib.convex_hull as convexhull
from matplotlib import pyplot as plt
from vision import init

X_RESOLUTION = 1080
Y_RESOLUTION = 720

#cap = init(X_RESOLUTION, Y_RESOLUTION)

cap = cv2.VideoCapture("hand_test/hands_test_1.mov")


while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    frame_gpu = cv2.UMat(frame)

    filtered_frame, mask = two_filter.filtered_frame(frame)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    contours = processing.filter_contours(contours)

    preHist = processing.hsv_histogram(colour.hsv(frame))

    postHist = processing.hsv_histogram(colour.hsv(filtered_frame))

    if len(contours) > 0:

        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        maxima, derivative, kcos_points = processing.contour_angle_maxima(max_contour, 30, 2)

        if len(kcos_points) > 0:
            for point in kcos_points:
                cv2.circle(frame_gpu, (point[0][0][0], point[0][0][1]), 10, (0, 255, 255))

        points = convexhull.get_hull_points(max_contour)
        convexhull.draw_hull_on_frame(frame_gpu, max_contour)
        cv2.drawContours(frame_gpu, [box], 0, (0, 0, 255), 1)
        cv2.drawContours(filtered_frame, [max_contour], -1, (255, 0, 0), 1, 1)

    cv2.imshow('Output Frame', cv2.resize(frame_gpu, None, fx=1, fy=1))
    cv2.imshow('Post Frame', cv2.resize(filtered_frame, None, fx=1, fy=1))
    cv2.imshow('Pre Hist', preHist)
    cv2.imshow('Post Hist', postHist)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
