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

    frame_gpu = cv2.UMat(frame)

    # TODO process the frame here
    returnFrame, contours, x, y, hull = test.test(frame)

    cv2.circle(frame, (int(x), int(y)), int(10), (0, 255, 255), 2)
    cv2.putText(frame, "point", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, 100)

    cv2.drawContours(frame, hull, -1, (255, 255, 255), 3, 8)

    hull_points = len(hull)
    point0 = hull[0][0][0]

    for i in range(0, hull_points):

        if i == hull_points - 1:
            cv2.line(frame, (hull[i][0][0], hull[i][0][1]), (hull[0][0][0], hull[0][0][1]), (255, 255, 255), 1)

        else:
            cv2.line(frame, (hull[i][0][0], hull[i][0][1]), (hull[i+1][0][0], hull[i+1][0][1]), (255, 255, 255), 1)

    cv2.drawContours(frame, contours, -1, (255, 255, 0), 1, 8)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
