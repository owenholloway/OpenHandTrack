# University of Tasmania
# Owen Holloway
# 206891
# 2019
import cv2
import alglib.test as test
import alglib.blur as blur

from vision import init

X_RESOLUTION = 1080
Y_RESOLUTION = 720

cap = init(X_RESOLUTION, Y_RESOLUTION)

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    frame_gpu = cv2.UMat(frame)



    returnFrame = test.test(frame)

    # Display the resulting frame
    cv2.imshow('frame', returnFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
