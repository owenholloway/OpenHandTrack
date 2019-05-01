# University of Tasmania
# Owen Holloway
# 206891
# 2019
import cv2
import numpy as np
import alglib.filter as filter
import alglib.processing as processing
import alglib.colour_space as colour

img1 = cv2.imread("hand_test/Hand_0001351.jpg")
img2 = cv2.imread("hand_test/Hand_0001108.jpg")

img2 = colour.white_balance(img2)

lower = np.array([0, 40, 0], np.uint8)
upper = np.array([255, 255, 255], np.uint8)

frame_blur1 = filter.guass(img1)
frame_sharp2 = filter.sharpen(img2)
frame_blur2 = filter.guass(img2)

img2_bw = colour.gray(frame_blur2)

contours1 = processing.contour_hsv(frame_blur1, lower, upper)

contours2 = processing.contour_canny(img2_bw, 0.44)

contours1_test = processing.filter_contours(contours1)

contours2_test = processing.filter_contours(contours2, 100)

img2 = cv2.drawContours(img2, contours2_test, -1, (255, 0, 255), 1, 8)

for cnt in contours2_test:
    img2 = cv2.fillPoly(img2, pts =[cnt], color=(0, 255, 0))

cv2.imshow('frame', img2)


while True:

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
