# University of Tasmania
# Owen Holloway
# 206891
# 2019
import cv2
import numpy as np
import alglib.filter as filter
import alglib.processing as processing
from os import listdir
from os.path import isfile, join
import time

path = "hand_test/"

hand_files = [f for f in listdir(path) if isfile(join(path, f))]

histograms = []

hands = 1

lower1 = np.array([0, 30, 0])
upper1 = np.array([180, 255, 256])

lower2 = np.array([25, 30, 0])
upper2 = np.array([160, 30, 256])

for hand in hand_files:

    img = cv2.imread(path+hand)

    img_blur = filter.guass(img, 0)

    hsv_filer1 = processing.hsv_mask(img_blur, lower1, upper1)
    hsv_filer2 = processing.hsv_mask(img_blur, lower2, upper2)
    cv2.bitwise_not(hsv_filer2, hsv_filer2)

    hsv_filter_final = cv2.bitwise_and(hsv_filer1, hsv_filer1)

    blocking_filter = cv2.bitwise_and(hsv_filer1, hsv_filer2)

    hsv_filter_final = cv2.bitwise_and(hsv_filer1, blocking_filter)
    img_masked = cv2.bitwise_and(img_blur, img_blur, mask=hsv_filter_final)

    img_canny = cv2.Canny(img_masked, 10, 40)

    mask = cv2.bitwise_and(hsv_filter_final, blocking_filter)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    max_contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    cv2.drawContours(img, [max_contour], -1, (255, 0, 0), 4, 8)
    cv2.imshow('frame' + str(hands), cv2.resize(img_canny, None, fx=0.5, fy=0.5))
    hands += 1

    if hands > 3:
        break


while True:

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
