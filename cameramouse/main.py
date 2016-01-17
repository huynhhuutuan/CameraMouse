import numpy as np
import cv2
import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from threading import Thread

__author__ = 'rbanalagay'


OFFSET = 200
COLOR = (0, 0, 255)
cap = cv2.VideoCapture(0)
img = np.ones((1920, 1080, 3), np.uint8) * 255
cv2.rectangle(img, (OFFSET, OFFSET), (OFFSET+495, OFFSET+495), COLOR, 10)
# cv2.rectangle(img, (OFFSET + 0, OFFSET + 0), (OFFSET + 5, OFFSET + 5), COLOR, -1)
# cv2.rectangle(img, (OFFSET + 490, OFFSET + 490), (OFFSET + 495, OFFSET + 495), COLOR, -1)
# cv2.rectangle(img, (OFFSET + 0, OFFSET + 490), (OFFSET + 5, OFFSET + 495), COLOR, -1)
# cv2.rectangle(img, (OFFSET + 490, OFFSET + 0), (OFFSET + 495, OFFSET + 5), COLOR, -1)
kernel = np.ones((5,5),np.uint8)


def start_camera():
    target_lines = 20
    min_vote_count = 70

    while True:
        ret, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # gray = frame[:, :, 2]
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # gray = clahe.apply(gray)
        mask = cv2.inRange(hsv, (160, 50, 50), (179, 255, 255))
        mask2 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        mask = cv2.add(mask, mask2)
        #
        # edges = cv2.Canny(gray,100,150)
        lines = cv2.HoughLines(mask, 1, (np.pi/180), round(min_vote_count))

        # lines = cv2.HoughLinesP(edges, 1, (np.pi/180), 70, 100, 0)
        # mask = cv2.medianBlur(mask, 3)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if lines is not None:
            if len(lines) < target_lines / 2:
                min_vote_count /= 2
            elif len(lines) > target_lines * 2:
                min_vote_count *= 2
            for line in lines:

                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                # x1, y1, x2, y2 = line[0]
                cv2.line(frame,(x1,y1),(x2,y2),(0,255,0), 2)

        #     print(len(lines))
        # else:
        #     min_vote_count /= 2


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

