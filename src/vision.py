import cv2


def init(xres, yres):

    print(cv2.__version__)

    print("starting vision")

    # change this number to set the camera being used
    cap = cv2.VideoCapture(0)

    cap.set(3, xres)
    cap.set(4, yres)

    return cap
