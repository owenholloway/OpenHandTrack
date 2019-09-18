import cv2
import numpy as np
import alglib.processing as processing
import alglib.filter as filter

lower1 = np.array([0, 40, 0])
upper1 = np.array([180, 255, 256])

lower2 = np.array([26, 0, 0])
upper2 = np.array([26, 45, 256])

lower3 = np.array([20, 0, 0])
upper3 = np.array([180, 256, 256])

lower4 = np.array([180, 125, 0])
upper4 = np.array([180, 256, 256])


def filtered_frame(frame):

    hsv_filer1 = processing.hsv_mask(frame, lower1, upper1)
    hsv_filer2 = processing.hsv_mask(frame, lower2, upper2)
    hsv_filer3 = processing.hsv_mask(frame, lower3, upper3)
    hsv_filer4 = processing.hsv_mask(frame, lower4, upper4)

    blocking = cv2.bitwise_or(hsv_filer3, hsv_filer4)

    cv2.bitwise_not(hsv_filer2, hsv_filer2)

    blocking_filter = cv2.bitwise_and(hsv_filer2, blocking)

    cv2.bitwise_not(blocking_filter, blocking_filter)

    hsv_filter_final = cv2.bitwise_and(hsv_filer1, blocking_filter)

    return cv2.bitwise_and(frame, frame, mask=hsv_filter_final), hsv_filter_final
