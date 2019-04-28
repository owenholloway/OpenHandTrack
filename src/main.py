# University of Tasmania
# Owen Holloway
# 206891
# 2019
import cv2
import alglib.test as test
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

    frame_blur = filter.sharpen(frame)

    max_contours_canny = processing.contour_canny(frame_blur)
    max_contours_hsv = processing.contour_hsv(frame_blur)

    if len(max_contours_canny) > 1:
        frame = cv2.drawContours(frame, max_contours_canny, -1, (255, 255, 0), 1, 8)
    if len(max_contours_hsv) > 1:
        frame = cv2.drawContours(frame, max_contours_hsv, -1, (0, 255, 255), 1, 8)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
