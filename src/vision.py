import cv2
import numpy as np

def init(xres, yres):

    print(cv2.__version__)

    print("starting vision")

    cap = cv2.VideoCapture(0)

    cap.set(3, xres)
    cap.set(4, yres)

    return cap
