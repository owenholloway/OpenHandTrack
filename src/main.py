# University of Tasmania
# Owen Holloway
# 206891
# 2019
import cv2
import numpy as np
import alglib.processing as processing
import alglib.two_filter as two_filter
import threading as threading
import alglib.convex_hull as convexhull
from vision import init

X_RESOLUTION = 1080
Y_RESOLUTION = 720

cap = init(X_RESOLUTION, Y_RESOLUTION)


def drawing_thread(frame_thread, box, max_contour, clusted_points):

    if len(max_contour) > 250:

        cv2.drawContours(frame_thread, [box], 0, (0, 0, 255), 2)
        #cv2.drawContours(frame_thread, [max_contour], -1, (255, 0, 0), 2)

        for i in range(0, len(clusted_points)):
            cv2.circle(frame_thread, (int(clusted_points[i][0][0][0]), int(clusted_points[i][0][0][1])), 6, (211, 0, 255), 3)

    frame_thread = frame_thread


first_run = True

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    filtered_frame, mask = two_filter.filtered_frame(frame)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    contours = processing.filter_contours(contours)

    #preHist = processing.hsv_histogram(colour.hsv(frame))

    #postHist = processing.hsv_histogram(colour.hsv(filtered_frame))

    if first_run:
        first_run = False
        frame_gpu = cv2.UMat(frame)
    else:
        drawing.join()
        cv2.imshow('Output Frame', cv2.resize(frame_gpu, None, fx=1, fy=1))
        frame_gpu = cv2.UMat(frame)

    if len(contours) > 0:

        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        points = convexhull.get_hull_points(max_contour)
        clusted_points = processing.contour_clustering(max_contour, points, 40)
        #convexhull.draw_hull_on_frame(frame_gpu, max_contour)

        drawing = threading.Thread(target=drawing_thread, args=(frame_gpu, box, max_contour, clusted_points))
        drawing.start()

    #cv2.imshow('Post Frame', cv2.resize(filtered_frame, None, fx=1, fy=1))
    #cv2.imshow('Pre Hist', preHist)
    #cv2.imshow('Post Hist', postHist)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

