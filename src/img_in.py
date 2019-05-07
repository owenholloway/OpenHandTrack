# University of Tasmania
# Owen Holloway
# 206891
# 2019
import cv2
import numpy as np
import alglib.filter as filter
import alglib.processing as processing
import alglib.colour_space as colour
from os import listdir
from os.path import isfile, join

path = "hands/"

hand_files = [f for f in listdir(path) if isfile(join(path, f))]

histograms = []

hands = 0

lower1 = np.array([0, 30, 0])
upper1 = np.array([180, 255, 256])

lower2 = np.array([25, 30, 0])
upper2 = np.array([160, 256, 256])


for hand in hand_files:
    img = cv2.imread(path+hand)

    img_blur = filter.guass(img, 0.8)

    hsv_filer1 = processing.hsv_mask(img_blur, lower1, upper1)
    hsv_filer2 = processing.hsv_mask(img_blur, lower2, upper2)
    cv2.bitwise_not(hsv_filer2, hsv_filer2)

    hsv_filter_final = cv2.bitwise_and(hsv_filer1, hsv_filer2)

    img_masked = cv2.bitwise_and(img, img, mask=hsv_filter_final)

    histograms.append(processing.hsv_histogram(colour.hsv(img_masked)))

    hands += 1

    if hands > 25:
        break

frame = 0

for hist in histograms:

    cv2.imshow('frame' + str(frame), hist)

    frame += 1


while True:

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
