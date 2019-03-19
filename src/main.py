# University of Tasmania
# Owen Holloway
# 206891
# 2019
import cv2
import alglib.test as test

from vision import init

XRESOLUTION = 1080
YRESOLUTION = 720

cap = init(XRESOLUTION, YRESOLUTION)

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # TODO process the frame here


    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
