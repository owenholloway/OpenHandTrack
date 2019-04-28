import cv2
import alglib.filters as filters
import alglib.convex_hull as ch


def test(frame):

    output_text = "Test text"

    cv2.putText(frame, output_text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    mask, frame_rtn = filters.hsv_mask(frame)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    (x, y) = (0, 0)

    if len(contours) > 0:

        c = max(contours, key=cv2.contourArea)

        hull_array, hull, defects = ch.get_hull_points(c)

        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if len(defects) > 0:

            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(c[s][0])
                end = tuple(c[e][0])
                far = tuple(c[f][0])
                cv2.line(frame_rtn, start, end, [0, 255, 0], 2)
                cv2.circle(frame_rtn, far, 5, [0, 0, 255], -1)

        cv2.circle(frame_rtn, (int(x), int(y)), int(10), (0, 255, 255), 2)
        cv2.putText(frame_rtn, "point", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, 100)

        hull_points = len(hull_array)
        if hull_points > 0:

            for i in range(0, hull_points):

                if i == hull_points - 1:
                    cv2.line(frame_rtn, (hull_array[i][0][0], hull_array[i][0][1]), (hull_array[0][0][0], hull_array[0][0][1]), (255, 255, 255), 1)

                else:
                    cv2.line(frame_rtn, (hull_array[i][0][0], hull_array[i][0][1]), (hull_array[i+1][0][0], hull_array[i+1][0][1]), (255, 255, 255), 1)

            cv2.drawContours(frame_rtn, contours, -1, (255, 255, 0), 1, 8)

    return frame_rtn
