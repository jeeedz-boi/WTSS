from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QAction, qApp, QMainWindow
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import face_recognition

import sys
import cv2
import numpy as np
import os
import time
from detection import Detection

from collections import deque

imageQueue = deque(maxlen=6)
feverImageQueue = deque(maxlen=1)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def imageScaler(self, img, scale=0.4):
        width = int(280 * scale)
        height = int(360 * scale)
        dim = (width, height)
        return cv2.resize(img, dim)

    # add new image to queue
    def OverlayUpdate(self, img):
        img = self.imageScaler(img)
        imageQueue.append(img)

    def OverlayFeverUpdate(self, fever_image):
        feverImageQueue.append(fever_image)

    
    # add 6 previous image to queue trigger only when app start
    def OverlayCreate(self):
        dlist = os.listdir(os.path.join('Storage/Normal'))
        d_ferver_list = os.listdir(os.path.join('Storage/Fever'))
        newlist = []
        for data in dlist:
            if ('.txt' in data or 'Display' in data):
                continue
            else:
                newlist.append(data)

        for data in sorted(newlist, reverse = True)[:6][::-1]:
            temp_image = cv2.imread(os.path.join('Storage/Normal/'+data), cv2.IMREAD_COLOR)
            fore = self.imageScaler(temp_image)
            imageQueue.append(fore)
        
        #exlastet_ferver = sorted(d_ferver_list, reverse = True)[0]
        #fever_image = cv2.imread(os.path.join('Storage/Fever/'+lastet_ferver[:-3]+'jpg'), cv2.IMREAD_COLOR)
        #feverImageQueue.append(fever_image)

    # set 6 images to be overlay
    def OverlaySet(self, back, offset, defaultHeight):
        queueSize = len(imageQueue)
        for index in range(queueSize):
            rows, cols, channels = imageQueue[queueSize-index-1].shape
            x, y = (cols*(index)+offset*(1+index), 720-rows-offset-10)
            back[y:y+rows, x:x+cols] = imageQueue[queueSize-index-1]

        #rows, cols, channels = feverImageQueue[0].shape
        #x, y = (1280-cols-offset, 720-rows-offset-10)
        #back[y:y+rows, x:x+cols] = feverImageQueue[0]
        return back
    
    def run(self):
        processing = Detection()
        # capture from web cam
        print('Application Start!')
        cap = cv2.VideoCapture(1)
        print('Video Capture Start!')
        print('Video Capture Setting initialize!')
        # Set video to 1920x1080
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print('Video Capture Setting initialize done! Video Stream Ready to start!')

        # OverlayCreate Call!
        self.OverlayCreate()
        print('Video Stream Start!')
        while self._run_flag:
            ret, cv_img = cap.read()

            cv_img, (is_past, detected_frame, temp) = processing.run(cv_img)
            
            if (is_past and temp < 37.5):
                self.OverlayUpdate(detected_frame)
            elif(is_past and temp >= 37.5):
                self.OverlayFeverUpdate(detected_frame)
            processed_frame = self.OverlaySet(cv_img, 10, 720)
            if ret:
                self.change_pixmap_signal.emit(processed_frame)
            else:
                self.stop()

        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()
        # self.setWindowTitle("Qt live label demo")
        self.disply_width = 1280
        self.display_height = 720

        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def Applicationclose():
        self.close()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("WFss")
        self.setGeometry(0, 0, 1280, 720)
        self.showFullScreen()

        self.app_widget = App()
        self.setCentralWidget(self.app_widget)

        exitAction = QAction('&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainApp = MainWindow()
    mainApp.show()
    sys.exit(app.exec_())
