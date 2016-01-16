import numpy as np
import cv2
import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from threading import Thread

__author__ = 'rbanalagay'


class OverlayWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.X11BypassWindowManagerHint
            )
        self.setGeometry(QtWidgets.QStyle.alignedRect(
            QtCore.Qt.LeftToRight, QtCore.Qt.AlignCenter,
            QtCore.QSize(220, 32),
            QtWidgets.qApp.desktop().availableGeometry()))

    def mousePressEvent(self, event):
        QtWidgets.qApp.quit()


OFFSET = 200
COLOR = (0, 0, 255)
cap = cv2.VideoCapture(0)
img = np.ones((1920, 1080, 3), np.uint8) * 255
cv2.rectangle(img, (300, 300), (800, 800), COLOR, 10)
# cv2.rectangle(img, (OFFSET + 0, OFFSET + 0), (OFFSET + 20, OFFSET + 20), COLOR, -1)
# cv2.rectangle(img, (OFFSET + 490, OFFSET + 490), (OFFSET + 510, OFFSET + 510), COLOR, -1)
# cv2.rectangle(img, (OFFSET + 0, OFFSET + 490), (OFFSET + 20, OFFSET + 510), COLOR, -1)
# cv2.rectangle(img, (OFFSET + 490, OFFSET + 0), (OFFSET + 510, OFFSET + 20), COLOR, -1)
kernel = np.ones((5,5),np.uint8)

def start_camera():
    while True:
        ret, frame = cap.read()

        mask = cv2.inRange(frame, (17, 15, 100), (100, 100, 255))

        lines = cv2.HoughLines(mask, 1, (np.pi/180), 100)
        print(lines)
        # mask = cv2.medianBlur(mask, 3)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        if lines is not None:
            for line in lines[0:10]:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    cv2.line(frame,(x1,y1),(x2,y2),(0,0,255), 2)


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

