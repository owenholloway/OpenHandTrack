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
from vision import init

X_RESOLUTION = 1080
Y_RESOLUTION = 720

cap = init(X_RESOLUTION, Y_RESOLUTION)

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    frame_gpu = cv2.UMat(frame)

    #frame_gpu = two_filter.filtered_frame(frame_gpu)

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
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

        points, hull, defects = convexhull.get_hull_points(max_contour)

        frame = cv2.drawContours(filtered_frame, contours, -1, (255, 0, 255), 1, 8)

    cv2.imshow('frame', filtered_frame)
    cv2.imshow('Pre Hist', preHist)
    cv2.imshow('Post Hist', postHist)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
