import cv2
import numpy as np
import alglib.processing as processing
import alglib.filter as filter

lower1 = np.array([0, 0, 0])
upper1 = np.array([180, 255, 256])

lower2 = np.array([25, 0, 0])
upper2 = np.array([160, 256, 256])


def filtered_frame(frame):

    img_blur = filter.guass(frame, 0.8)

    hsv_filer1 = processing.hsv_mask(img_blur, lower1, upper1)
    hsv_filer2 = processing.hsv_mask(img_blur, lower2, upper2)
    cv2.bitwise_not(hsv_filer2, hsv_filer2)

    hsv_filter_final = cv2.bitwise_and(hsv_filer1, hsv_filer2)

    return cv2.bitwise_and(frame, frame, mask=hsv_filter_final), hsv_filter_final
