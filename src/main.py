# University of Tasmania
# Owen Holloway
# 206891
# 2019
import cv2
import numpy as np
import alglib.filter as filter
import alglib.processing as processing
import alglib.colour_space as colour

from vision import init

X_RESOLUTION = 1080
Y_RESOLUTION = 720

cap = init(X_RESOLUTION, Y_RESOLUTION)

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    frame_gpu = cv2.UMat(frame)

    frame_blur = filter.guass(frame)
    frame_blur_gray = colour.gray(frame_blur)

    frame_sharpen = filter.sharpen(frame)
    frame_sharpen_gray = colour.hsv(frame_sharpen)
    frame_sharpen_gray_blur = filter.guass(frame)

    frame_gpu_blur = filter.guass(frame_gpu)

    #contours_canny_blur = processing.contour_canny(frame_blur_gray)
    #contours_canny_sharp = processing.contour_canny(frame_sharpen_gray_blur)

    # contours_hsv = processing.contour_hsv(frame_gpu_blur)

    # blobs = processing.blob_detect(frame_gpu_blur)

    # if len(contours_hsv) > 1:
    #    frame_gpu = cv2.drawContours(frame_gpu, contours_hsv, -1, (0, 255, 255), 1, 8)

    # if len(contours_canny_blur) > 1:
        # frame_gpu = cv2.drawContours(frame_gpu, contours_canny_blur, -1, (255, 255, 0), 1, 8)

    #if len(contours_canny_sharp) > 1:
    #    frame_gpu = cv2.drawContours(frame_gpu, contours_canny_sharp, -1, (255, 0, 255), 1, 8)

    # frame_gpu = cv2.drawContours(frame_gpu, blobs, -1, (255, 255, 0), 1, 8)

    # Display the resulting frame
    cv2.imshow('frame', frame_gpu)

    cv2.imshow('hist', processing.hsv_histogram(frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
