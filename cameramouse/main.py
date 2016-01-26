import numpy as np
import cv2
import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from threading import Thread

__author__ = 'rbanalagay'


OFFSET = 200
COLOR = (0, 0, 255)
BOX_SIZE = 500
CIRCLE_RADIUS = 30

cap = cv2.VideoCapture(0)
img = np.ones((1920, 1080, 3), np.uint8) * 255


BOX_CORNERS = ((OFFSET, OFFSET), (OFFSET + BOX_SIZE, OFFSET),
               (OFFSET, OFFSET + BOX_SIZE),
               (OFFSET + BOX_SIZE, OFFSET + BOX_SIZE))
for corner in BOX_CORNERS:
    cv2.circle(img, corner, CIRCLE_RADIUS, COLOR, -1)

KERNEL = np.ones((5, 5), np.uint8)


def start_camera():

    while True:
        ret, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, (160, 50, 50), (179, 255, 255))
        mask2 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        mask = cv2.add(mask, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)

        __, circles, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)
        circles = sorted(circles, key=cv2.contourArea, reverse=True)[:4]
        for circle in circles:
            M = cv2.moments(circle)
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])
            cv2.drawContours(frame, [circle], -1, (0, 255, 0), 3)
            cv2.circle(frame, (centroid_x, centroid_y), 3, (0, 255, 255), -1)


        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow('overlay', img)
        # cv2.moveWindow('overlay', 0, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

start_camera()
# camera_thread = Thread(target=start_camera)
# camera_thread.start()

# app = QtWidgets.QApplication(sys.argv)
# overlay_window = OverlayWindow()
# overlay_window.show()
# overlay_thread = Thread(target=app.exec_())
# overlay_thread.start()

# camera_thread.join()

