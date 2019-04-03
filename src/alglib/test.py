import cv2
import numpy as np

def test(frame):

    output_text = "Test text"

    cv2.putText(frame, output_text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    edges = cv2.Canny(frame, 8, 60)

    colour_space = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    #lower = np.array([2, 35, 128], np.uint8)
    #upper = np.array([30, 124, 255], np.uint8)

    #mask = cv2.inRange(hsv, lower, upper)
    #mask = cv2.erode(mask, None, iterations=2)
    #mask = cv2.dilate(mask, None, iterations=2)

    #contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    (x, y) = (0, 0)

    #c = max(contours, key=cv2.contourArea)
    #((x, y), radius) = cv2.minEnclosingCircle(c)

    return colour_space, x, y
