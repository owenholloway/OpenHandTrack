import cv2
import numpy as np
import alglib.processing as processing

lower1 = np.array([0, 0, 0])
upper1 = np.array([180, 250, 256])


def filtered_frame(frame):

    hsv_filer1 = processing.hsv_mask(frame, lower1, upper1)

    hsv_filter_final = cv2.bitwise_and(hsv_filer1, hsv_filer1)

    return cv2.bitwise_and(frame, frame, mask=hsv_filter_final), hsv_filter_final
