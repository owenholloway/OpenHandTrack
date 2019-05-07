# University of Tasmania
# Owen Holloway
# 206891
# 2019
import cv2
import numpy as np
import alglib.filter as filter
import alglib.processing as processing
import alglib.colour_space as colour
import alglib.two_filter as two_filter

from vision import init

X_RESOLUTION = 1080
Y_RESOLUTION = 720

cap = init(X_RESOLUTION, Y_RESOLUTION)

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    frame_gpu = cv2.UMat(frame)

    frame_gpu = two_filter.filtered_frame(frame_gpu)

    filtered_frame = two_filter.filtered_frame(frame)

    pre_canny = filter.sharpen(filtered_frame)
    pre_canny = colour.gray(pre_canny)

    canny_contours = processing.contour_canny(pre_canny)

    if len(canny_contours) > 0:
        frame_gpu = cv2.drawContours(frame_gpu, canny_contours, -1, (255, 0, 255), 1, 8)


    cv2.imshow('frame', frame_gpu)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
