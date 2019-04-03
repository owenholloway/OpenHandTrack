# University of Tasmania
# Owen Holloway
# 206891
# 2019
import cv2
import alglib.test as test

from vision import init

X_RESOLUTION = 1080
Y_RESOLUTION = 720

cap = init(X_RESOLUTION, Y_RESOLUTION)

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    # TODO process the frame here
    returnFrame, x, y = test.test(frame)

    #cv2.circle(returnFrame, (int(x), int(y)), int(10), (0, 255, 255), 2)
    #cv2.putText(frame, "point", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, 100)

    # Display the resulting frame
    cv2.imshow('frame', returnFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
