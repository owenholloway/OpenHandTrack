import cv2


def test(frame):

    output_text = "Test text"
    cv2.putText(frame, output_text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    edges = cv2.Canny(frame, 0, 75)
    find_hand_centre(frame, edges)
    return edges


def find_hand_centre(frame, contours):
    mask = cv2.inRange(contours, 1, 100)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    x = 0
    y = 0

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        # print(M)
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(contours, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(contours, center, 5, (0, 0, 255), -1)
            text = "X:%i Y:%i" % (x, y)
            cv2.putText(contours, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, 100)
            x = x
            y = y

    return x, y
