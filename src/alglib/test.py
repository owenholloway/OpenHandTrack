import cv2
import alglib.filters as filters
import alglib.convex_hull as ch

def test(frame):

    output_text = "Test text"

    cv2.putText(frame, output_text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    mask = filters.hsv_mask(frame)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    (x, y) = (0, 0)

    c = max(contours, key=cv2.contourArea)

    hull, defects = ch.get_hull_points(c)

    ((x, y), radius) = cv2.minEnclosingCircle(c)

    return mask, contours, x, y, hull
