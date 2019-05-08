# University of Tasmania
# Owen Holloway
# 206891
# 2019
import cv2
import alglib.processing as processing
import alglib.two_filter as two_filter
import alglib.convex_hull as convexhull
from vision import init

X_RESOLUTION = 1080
Y_RESOLUTION = 720

cap = init(X_RESOLUTION, Y_RESOLUTION)

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    frame_gpu = cv2.UMat(frame)

    frame_gpu = two_filter.filtered_frame(frame_gpu)

    filtered_frame, mask = two_filter.filtered_frame(frame)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    contours = processing.filter_contours(contours)

    for cnt in contours:
        hull = convexhull.get_hull_points(cnt)
        hull_points = len(hull)
        if len(hull) > 0:
            cv2.drawContours(frame, hull, -1, (255, 0, 255), 1, 8)
    #    point0 = hull[0][0][0]
    #    if len(hull) > 1:
    #        for i in range(0, hull_points):

    #            if i == hull_points - 1:
    #                cv2.line(frame, (hull[i][0][0], hull[i][0][1]), (hull[0][0][0], hull[0][0][1]), (255, 255, 255), 1)

     #           else:
      #              cv2.line(frame, (hull[i][0][0], hull[i][0][1]), (hull[i + 1][0][0], hull[i + 1][0][1]), (255, 255, 255), 1)

    #pre_canny = filter.sharpen(filtered_frame)
    #pre_canny = colour.gray(pre_canny)

    #canny_contours = processing.contour_canny(filtered_frame)

    if len(contours) > 0:
        frame = cv2.drawContours(frame, contours, -1, (255, 0, 255), 1, 8)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
