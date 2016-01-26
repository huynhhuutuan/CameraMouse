import numpy as np
import cv2
import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from threading import Thread
from pymouse import PyMouse

__author__ = 'rbanalagay'

MOUSE = PyMouse()
OFFSET = 200
COLOR = (0, 0, 255)
BOX_SIZE = 500
CIRCLE_RADIUS = 30

cap = cv2.VideoCapture(0)


BOX_CORNERS = ((OFFSET, OFFSET), (OFFSET + BOX_SIZE, OFFSET),
               (OFFSET, OFFSET + BOX_SIZE),
               (OFFSET + BOX_SIZE, OFFSET + BOX_SIZE))


KERNEL = np.ones((10, 10), np.uint8)


def sort_corners(corners):
    center = np.zeros(2)
    for corner in corners:
        center += np.array(corner)
    center /= 4.0

    top_points = []
    bottom_points = []
    for corner in corners:
        x, y = corner
        if y < center[1]:
            bottom_points.append(corner)
        else:
            top_points.append(corner)

    top_left = (top_points[0]
                if top_points[0][0] < top_points[1][0]
                else top_points[1])
    top_right = (top_points[0]
                 if top_points[0][0] >= top_points[1][0]
                 else top_points[1])
    bottom_left = (bottom_points[0]
                   if bottom_points[0][0] < bottom_points[1][0]
                   else bottom_points[1])
    bottom_right = (bottom_points[0]
                    if bottom_points[0][0] >= bottom_points[1][0]
                    else bottom_points[1])
    return np.array(top_left), np.array(top_right), np.array(bottom_left), np.array(bottom_right)


def start_camera():

    while True:
        ret, frame = cap.read()
        img = np.ones((1080, 1920, 3), np.uint8) * 255
        for corner in BOX_CORNERS:
            cv2.circle(img, corner, CIRCLE_RADIUS, COLOR, -1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, (160, 50, 50), (179, 255, 255))
        mask2 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        mask = cv2.add(mask, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)

        __, circles, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)
        circles = sorted(circles, key=cv2.contourArea, reverse=True)[:4]
        corners = []
        for circle in circles:
            M = cv2.moments(circle)
            centroid_x = M['m10']/M['m00']
            centroid_y = M['m01']/M['m00']
            corners.append((centroid_x, centroid_y))
            cv2.drawContours(frame, [circle], -1, (0, 255, 0), 3)
            cv2.circle(frame, (int(centroid_x), int(centroid_y)),
                       3, (0, 255, 255), -1)

        if len(circles) == 4:
            sorted_corners = np.array(
                np.vstack(sort_corners(corners)),
                dtype='float32')
            box_corners = np.array(
                np.vstack([np.array(corner) for corner in BOX_CORNERS]),
                dtype='float32')

            transform_matrix = cv2.getPerspectiveTransform(sorted_corners, box_corners)
            height, width, channels = frame.shape
            pt = transform_matrix.dot(np.array([width / 2.0, height / 2.0, 1]))
            cv2.circle(img, (int(pt[0]), int(1080 - pt[1])), 3, (0,0,0), -1)
            # print(int(pt[0]), int(pt[1]))
            # MOUSE.move(int(pt[0]), int(pt[1]))


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

