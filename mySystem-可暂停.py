# -- coding:utf-8 --
# author: ZQF time:2018/12/15


import sys
import time
import cv2
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui

from PyQt5.QtWidgets import QMainWindow, QApplication, QSplashScreen, QFileDialog,\
     QMessageBox, QDesktopWidget, QGraphicsScene, QDialog
from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from car_interface import Ui_MainWindow


class Change_pixmap():
    def cv2_rgb_to_pixmap(self, img, label=None):
        # 将cv2中 rgb 格式的图片转换为QPixmap能作用的格式，并设置图片尺寸为label控件的大小
        tul = 3 * img.shape[1]  # 这个地方很奇怪，至今不能理解
        pic_q_image = QImage(img.data, img.shape[1], img.shape[0], tul,
                                      QImage.Format_RGB888)
        if label is None:
            pic_pixmap = QPixmap.fromImage(pic_q_image)
        else:
            pic_pixmap = QPixmap.fromImage(pic_q_image).scaled(label.width(), label.height())
        return pic_pixmap

    def cv2_gray_to_pixmap(self, img, label=None):
        # 将cv2中 gray 格式的图片转换为QPixmap能作用的格式，并设置图片尺寸为label控件的大小
        tul = 1 * img.shape[1]  # 这个地方很奇怪，至今不能理解
        pic_q_image = QImage(img.data, img.shape[1], img.shape[0], tul,QImage.Format_Grayscale8)
        if label is None:
            pic_pixmap = QPixmap.fromImage(pic_q_image)
        else:
            pic_pixmap = QPixmap.fromImage(pic_q_image).scaled(label.width(), label.height())
        return pic_pixmap


class CarMain(QMainWindow, Ui_MainWindow, Change_pixmap):  # 继承了上面一个类，方便使用一些功能
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.connections()
        # pic = QPixmap('E:\\have\Wall-E.jpg').scaled(self.label_frameBox.width(), self.label_frameBox.height())
        # self.label_frameBox.setPixmap(pic)
        self.x1, self.x2, self.y1, self.y2 = None, None, None, None
        self.count = 1
        self.th = None
        # self.pushButton_play_pause.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

    def connections(self):
        self.action_open.triggered.connect(self.open_video)
        self.action_rotate.triggered.connect(self.rotate_frame)
        self.action_viewCap.triggered.connect(self.select_view)
        self.action_dataCap.triggered.connect(self.select_data)
        self.label_frameBox.mousePressEvent = self.get_points
        self.label_frameBox.mouseReleaseEvent = self.get_points

    def open_video(self):
        global video_path
        video_path, video_type = QFileDialog.getOpenFileName(self, "选择视频", "G:\Python\无人机车辆识别\探索阶段",
                                                             "*.mov *.wmv *.mp4;;All Files(*)")
        if video_path == '':
            return
        else:
            self.th = Thread(self)
            self.th.send_pic.connect(self.show_frame)
            self.th.send_time.connect(self.set_time)
            self.horizontalSlider.sliderReleased.connect(lambda: self.th.change_picShow(2000))
            self.pushButton_play_pause.clicked.connect(self.th.play_pause)
            self.th.start()

    def show_frame(self, image, ms):  # 将传来的cv2格式图片转换为Qpixmap的并显示在label中
        pixmap = self.cv2_rgb_to_pixmap(image, self.label_frameBox)
        self.label_frameBox.setPixmap(pixmap)
        self.horizontalSlider.setValue(ms)  # 让slider滑块随视频播放而移动

    def set_time(self, time):
        print('total_frame:', time)
        self.horizontalSlider.setMaximum(time)  # 设置slider的最大值

    def rotate_frame(self):
        if self.action_rotate.isChecked():
            self.x1, self.x2, self.y1, self.y2 = None, None, None, None
            self.statusBar.showMessage('请绘制车道线', 0)
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
            if self.x1 is None:
                QMessageBox.information(self, '提示', '请绘制车道线', QMessageBox.Ok)
            else:
                img = cv2.imread('E:\\have\Wall-E.jpg')

                theta = cv2.fastAtan2((self.y2 - self.y1), (self.x2 - self.x1))
                h, w, c = img.shape
                m = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1)
                pic1 = cv2.warpAffine(img, m, (w, h))
                rgb = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
                show = self.cv2_rgb_to_pixmap(rgb, self.label_frameBox)
                self.label_frameBox.setPixmap(show)

    def select_view(self):
        if self.action_viewCap.isChecked():
            self.x1, self.x2, self.y1, self.y2 = None, None, None, None
            self.statusBar.showMessage('请选择视区', 0)
            self.setCursor(Qt.CrossCursor)
            # cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), (255, 100, 0), 2)
        else:
            self.setCursor(Qt.ArrowCursor)
            img = cv2.imread('E:\\have\Wall-E.jpg')
            img = cv2.resize(img, (self.label_frameBox.width(), self.label_frameBox.height()), interpolation=cv2.INTER_CUBIC)
            if self.x1 is None:
                QMessageBox.information(self, '提示', '请选择区域', QMessageBox.Ok)
            else:
                if self.x1 > self.x2:
                    self.x1, self.x2 = self.x2, self.x1
                if self.y1 > self.y2:
                    self.y1, self.y2 = self.y2, self.y1
                if self.y1 < 0:
                    self.y1 = 0
                elif self.y2 > self.label_frameBox.height():
                    self.y2 = self.label_frameBox.height()
                if self.x1 < 0:
                    self.x1 = 0
                elif self.x2 > self.label_frameBox.width():
                    self.x2 = self.label_frameBox.width()
                img = img[self.y1:self.y2, self.x1:self.x2, :]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.show = self.cv2_rgb_to_pixmap(rgb)
                self.label_frameBox.setPixmap(self.show)

    def select_data(self):
        pass

    def get_points(self, event):
        if self.count == 1:
            self.x1 = event.pos().x()
            self.y1 = event.pos().y()
            print('获得点坐标1：', event.pos().x(), event.pos().y())
            self.count = 2
        elif self.count == 2:
            self.x2 = event.pos().x()
            self.y2 = event.pos().y()
            print('获得点坐标2：', event.pos().x(), event.pos().y())
            self.count = 1
            self.statusBar.showMessage('', 0)

    def resizeEvent(self, event):
        self.label_frameBox.setGeometry(QtCore.QRect(0, 0, event.size().width(), event.size().height()-20))
        pic = QPixmap('E:\\have\Wall-E.jpg').scaled(self.label_frameBox.width(), self.label_frameBox.height())
        self.label_frameBox.setPixmap(pic)
        self.update()


class Thread(QThread):
    send_pic = pyqtSignal(np.ndarray, float)
    send_time = pyqtSignal(int)
    pp = True

    def run(self):
        self.cap = cv2.VideoCapture()
        self.cap.open(video_path)
        print(video_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)  # 帧率
        frame_num = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获得视频总帧数
        frame_time = int(frame_num/fps*1000)  # 视频总毫秒数
        self.send_time.emit(frame_time)  # 将视频总毫秒数传回
        while self.cap.isOpened():
            if self.pp:  # 使用这个标识来限制read cap，每次read之后当前帧数会自动加1
                ret, frame = self.cap.read()  # 感觉是这里出的问题。没错，就是这个的问题
                if ret:
                    print(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    frame_cv2_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cur_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)  # 当前毫秒数
                    self.send_pic.emit(frame_cv2_rgb, cur_ms)  # 将当前帧的画面和帧数传回
                    # time.sleep(0.01)  # 控制视频播放的速度,似乎不需要
                else:
                    break
            else:
                pass
        self.cap.release()

    def change_picShow(self, picshow):
        print(picshow)
        print(type(picshow))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, picshow)

    def play_pause(self):
        print('pause')
        self.pp = not self.pp
        print('ok')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    car_main = CarMain()
    car_main.show()
    app.exec_()
