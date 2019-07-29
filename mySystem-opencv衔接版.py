# -- coding:utf-8 --
# author: ZQF time:2018/12/15


import sys
import time
import cv2
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog,\
     QMessageBox, QDesktopWidget, QGraphicsScene, QDialog, QTableWidgetItem
from PyQt5.QtCore import pyqtSlot, Qt, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QPixmap, QImage

from car_interface import Ui_MainWindow
import ColorThresholder, ContourFilter

'''
这个是保存的最接近opencv版的版本
'''


class ColorThreshFilter:
    def __init__(self):
        self.cars = {}
        self.Lanes, bkg, bk_cnts = self.bkg_process()
        self.cnts = None
        self.h_min = 0.388 * 180  # 70
        self.h_max = 0.021 * 180  # 4
        self.s_min = 0.000 * 256  # 0
        self.s_max = 0.116 * 256  # 30
        self.v_min = 0.398 * 256  # 102
        self.v_max = 0.776 * 256  # 199

    def adjust_frame(self, frame):
        rows, cols, ch = frame.shape
        M = cv2.getRotationMatrix2D((cols, rows), 1, 1)  # 三个参数分别是旋转中心，旋转角度，比例
        frame = cv2.warpAffine(frame, M, (cols, rows))
        frame = frame[580:670, 470:1030]
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        return frame

    def bkg_process(self):
        bkg = cv2.imread('bkg_gap20_cur1179.0.jpg', 0)  # 城市道路背景图
        bkg = cv2.GaussianBlur(bkg, (5, 5), 0)
        _, bkg = cv2.threshold(bkg, 190, 255, cv2.THRESH_BINARY)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        bkg = cv2.dilate(bkg, kernel1, iterations=1)
        _, contours, hierarchy = cv2.findContours(bkg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Lanes = {}
        for i, cnt in enumerate(contours):
            c = self.calc_characters(cnt)
            Lanes[i] = c
        return Lanes, bkg, contours  # 其实已经不需要这个返回的bkg了

    def get_mask(self, bgr):
        I = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        if self.h_min >= self.h_max:
            mask_h = (I[:, :, 0] >= self.h_min) | (I[:, :, 0] <= self.h_max)
        else:
            mask_h = (I[:, :, 0] >= self.h_min) & (I[:, :, 0] <= self.h_max)
        if self.s_min >= self.s_max:
            mask_s = (I[:, :, 1] >= self.s_min) | (I[:, :, 1] <= self.s_max)
        else:
            mask_s = (I[:, :, 1] >= self.s_min) & (I[:, :, 1] <= self.s_max)
        if self.v_min >= self.v_max:
            mask_v = (I[:, :, 2] >= self.v_min) | (I[:, :, 2] <= self.v_max)
        else:
            mask_v = (I[:, :, 2] >= self.v_min) & (I[:, :, 2] <= self.v_max)
        mask = ~(mask_h & mask_s & mask_v) * 255
        mask = np.uint8(mask)
        masked_pic = cv2.bitwise_and(bgr, bgr, mask=mask)
        return mask, masked_pic

    def calc_characters(self, cnt):
        moments = cv2.moments(cnt)
        m00 = moments['m00']  # 面积
        m10 = moments['m10']
        m01 = moments['m01']
        if m00 == 0:
            x, y = 0, 0  # 随便定个坐标
        else:
            x = int(m10 / m00)  # 重心横坐标
            y = int(m01 / m00)  # 重心纵坐标
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(cnt)  # 获得轮廓外接矩形的左上角点坐标、宽、高
        EquivDiameter = np.sqrt(4 * m00 / np.pi)  # 与轮廓面积相等的圆的直径
        rect = cv2.minAreaRect(cnt)  # 得到轮廓最小外接矩形的中心，宽高和旋转角度
        Orientation = rect[2]
        if rect[1][0] * rect[1][1] > 0:  # 经过开运算后好像最小外接矩形的面积可能等于0
            Extent = m00 / (rect[1][0] * rect[1][1])  # 轮廓和最小外接矩形的面积比
        else:
            Extent = 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = m00 / hull_area  # 轮廓和凸包面积的比
        else:
            solidity = 0
        try:
            ellipse = cv2.fitEllipse(cnt)  # 最小外切矩形的内切椭圆，返回值为（中心），（短长轴），旋转角度
        except:
            Eccentricity = 0
        else:
            a = max(ellipse[1]) / 2
            b = min(ellipse[1]) / 2
            Eccentricity = np.sqrt(a ** 2 - b ** 2) / a  # 离心率
        L = tuple(cnt[cnt[:, :, 0].argmin()][0])
        R = tuple(cnt[cnt[:, :, 0].argmax()][0])

        T = tuple(cnt[cnt[:, :, 1].argmin()][0])
        B = tuple(cnt[cnt[:, :, 1].argmax()][0])
        characters = {'area': m00, 'center': [x, y], 'EquivDiameter': EquivDiameter, 'Extent': Extent,
                      'solidity': solidity, 'Eccentricity': Eccentricity, 'Orientation': Orientation,
                      'rect_x': rect_x, 'rect_y': rect_y, 'rect_w': rect_w, 'rect_h': rect_h,
                      'L': L, 'R': R, 'T': T, 'B': B}
        return characters

    def cnt_filter(self, contours, Lanes, mask):
        for cnt in contours:
            C = self.calc_characters(cnt)
            x, y, w, h = C['rect_x'], C['rect_y'], C['rect_w'], C['rect_h']
            if C['area'] < 20 or (h < 15 and w > 20):  # 面积小于20的和长得像车道线的直接过滤不要了
                # cv2.rectangle(frame_color, (x, y), (x+w, y+h), (0, 0, 255), -1)
                cv2.drawContours(mask, [cnt], 0, (0, 0, 0), -1)  # 把该轮廓的像素值全部设置为0
                continue
            # for char in CHARS:
            for line in Lanes.values():  # Lanes是个字典，里面是从背景图里面提取到的轮廓对象
                # 如果在Lanes里面找到了轮廓和该轮廓对应
                if -30 < line['center'][0] - C['center'][0] < 30 and -10 < line['center'][1] - C['center'][1] < 10 and \
                        C['area'] < 100:
                    # cv2.rectangle(frame_color, (x, y), (x+w, y+h), (0, 0, 255), -1)
                    cv2.drawContours(mask, [cnt], 0, (0, 0, 0), -1)  # 把整个轮廓的像素值全部设置为0
                    break
                # 如果是和车辆粘在一起的车道线
                if -30 < (line['L'][0] - C['L'][0]) < 30 and -20 < (line['L'][1] - C['L'][1]) < 20:
                    bool_l = (C['L'][0] <= cnt[:, 0, 0]) & (cnt[:, 0, 0] <= C['L'][0] + 15) & (
                                C['L'][1] - 5 <= cnt[:, 0, 1]) & (cnt[:, 0, 1] <= C['L'][1] + 5)
                    slice_l = cnt[bool_l]
                    if len(slice_l):
                        cv2.drawContours(mask, [slice_l], 0, (0, 0, 0), -1)
                        # break
                    else:
                        cv2.waitKey(0)  # 只是留着检查用，如果出现意外会停下来

                if -30 < (line['R'][0] - C['R'][0]) < 30 and -20 < (line['R'][1] - C['R'][1]) < 20:
                    bool_r = (C['R'][0] - 15 <= cnt[:, 0, 0]) & (cnt[:, 0, 0] <= C['R'][0]) & (
                                C['R'][1] - 5 <= cnt[:, 0, 1]) & (cnt[:, 0, 1] <= C['R'][1] + 5)
                    slice_r = cnt[bool_r]
                    if len(slice_r):
                        cv2.drawContours(mask, [slice_r], 0, (0, 0, 0), -1)
                        # break
                    else:
                        cv2.waitKey(0)  # 只是留着检查用，如果出现意外会停下来
        return mask

    def draw_and_text(self, cur, mask, frame_color):
        Cs = []
        _, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = cnts  # 弄成实例属性，方便在playerBox里面调用
        for cnt in cnts:
            C = self.calc_characters(cnt)
            Cs.append(C)
            x, y, w, h = C['rect_x'], C['rect_y'], C['rect_w'], C['rect_h']
            # cx, cy = C['center']
            # 过滤完面积较小的轮廓和车道线之后，如果轮廓是个车
            if (0 < C['Eccentricity'] < 0.96 and C['area'] > 100) or C['area'] > 200:
                if cur == 0:  # 如果是第一帧
                    print(1, len(self.cars))
                    self.cars[len(self.cars)] = (x, y, w, h)
                    # self.cars[len(self.cars)] = (cx, cy)
                    cv2.rectangle(frame_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)
                    cv2.putText(frame_color, str(len(self.cars)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                                cv2.LINE_AA)
                else:  # 如果不是第一帧
                    found = False
                    if 5 < x < 830 and 5 < y < 120:
                        for each in self.cars:  # 遍历储存的车辆，查找是否有与该轮廓邻近的信息，如果有，则认为该轮廓即搜索到的车辆
                            if -40 < x - self.cars[each][0] < 40 and -30 < y - self.cars[each][1] < 30:
                                self.cars[each] = (x, y, w, h)
                                # self.cars[each] = (cx, cy)
                                cv2.rectangle(frame_color, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 在frame上绘制矩形
                                # cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)
                                cv2.putText(frame_color, str(each), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                                            2, cv2.LINE_AA)
                                found = True  # 如果找到了，设置一个flag
                        if not found:  # 如果没有找到，则把该轮廓标记为新的车辆
                            print(2, len(self.cars))
                            self.cars[len(self.cars)] = (x, y, w, h)
                            cv2.rectangle(frame_color, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 在frame上绘制矩形
                            cv2.putText(frame_color, str(len(self.cars)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (255, 255, 0), 2, cv2.LINE_AA)
            else:
                # cv2.rectangle(frame_color, (x, y), (x+w, y+h), (0, 0, 255), -1)
                cv2.drawContours(mask, [cnt], 0, (0, 0, 0), -1)  # 把该轮廓的像素值全部设置为0
        return mask, frame_color, Cs

    def main_proceed(self, frame, cur_num):
        frame = self.adjust_frame(frame)  # 把视频调整为合适的大小 城市道路视频
        # frame_color = adjust_frame_highway(frame)  # 高速路视频

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        adp_pic = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 6)

        frame_color = frame.copy()
        mask = cv2.GaussianBlur(frame_color, (3, 3), 0)
        frame_adjusted = frame.copy()  # 备份一个，用来返回后传入ColorThresholder里面查看阈值设置的效果
        mask, pic = self.get_mask(mask)

        # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # mask = cv2.erode(mask, kernel2, iterations=1)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2, iterations=1)

        # mask = cv2.bitwise_or(mask, adp_pic)  # 将局部阈值处理的结果和颜色过滤后的结果取了并集，目前看来效果不太理想
        # mask = do_morph(mask)
        bw = mask.copy()  # 复制一份，传回，用来在PlayerBox里面显示轮廓筛选之前的效果
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = self.cnt_filter(contours, self.Lanes, mask)
        mask, frame_color, mask_Cs = self.draw_and_text(cur_num, mask, frame_color)
        return frame_color, bw, mask, mask_Cs, frame_adjusted


class PlayerBox(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.connections()
        # pic = QPixmap('E:\\have\Wall-E.jpg').scaled(self.label_frameBox.width(), self.label_frameBox.height())
        # self.label_frameBox.setPixmap(pic)
        self.x1, self.x2, self.y1, self.y2 = None, None, None, None
        self.count = 1
        self.th = None
        self.frame_color, self.bw, self.mask = None, None, None
        self.frame_adjusted = None
        self.CTF = ColorThreshFilter()  # 实例化，调用专门处理车辆识别的类
        # self.pushButton_play_pause.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

    def connections(self):
        self.action_open.triggered.connect(self.open_video)
        self.action_rotate.triggered.connect(self.rotate_frame)
        self.action_viewCap.triggered.connect(self.select_view)
        self.action_dataCap.triggered.connect(self.select_data)
        self.label_frameBox.mousePressEvent = self.get_points
        self.label_frameBox.mouseReleaseEvent = self.get_points
        self.tableWidget.itemClicked.connect(self.highlight_cars)
        self.action_ColorThreshholder.triggered.connect(self.open_thresholder)
        self.action_ContourFilter.triggered.connect(self.openfilter)

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
            self.pushButton_play_pause.clicked.connect(lambda: self.th.play_pause(self.pushButton_play_pause.isChecked()))
            print('self.th.start()')
            self.th.start()

    def show_frame(self, frame_bgr, cur_ms, cur_num):  # 将传来的cv2格式图片转换为Qpixmap的并显示在label中
        # jpg的可以，png的不行
        self.frame_color, self.bw, self.mask, mask_Cs, self.frame_adjusted = self.CTF.main_proceed(frame_bgr, cur_num)

        self.show_table(self.tableWidget, mask_Cs)  # 显示右侧表格数据。Cs是储存了当前过滤完后一帧图像中所有轮廓所有特诊的一个列表

        # 显示这三张图片
        self.set_label_pic(self.frame_color, self.bw, self.mask)

        self.horizontalSlider.setValue(cur_ms)  # 让slider滑块随视频播放而移动

    def set_label_pic(self, frame_bgr, bw, mask):
        pixmap_color = self.cv2_pic_to_pixmap(frame_bgr, 'bgr')  # 如果做切片，这句将无法调用
        pixmap_bw = self.cv2_pic_to_pixmap(bw, 'gray')
        pixmap_mask = self.cv2_pic_to_pixmap(mask, 'gray')
        self.label_frameBox.setPixmap(pixmap_color)
        self.label2.setPixmap(pixmap_bw)
        self.label3.setPixmap(pixmap_mask)

    def show_table(self, table, Cs):
        table.setRowCount(len(Cs))
        for k, C in enumerate(Cs):
            table.setItem(k, 0, QTableWidgetItem(str(C['area'])))
            table.setItem(k, 1, QTableWidgetItem(str(C['center'])))
            table.setItem(k, 2, QTableWidgetItem(str(C['Eccentricity'])))
            table.setItem(k, 3, QTableWidgetItem(str(C['EquivDiameter'])))
            table.setItem(k, 4, QTableWidgetItem(str(C['Extent'])))
            table.setItem(k, 5, QTableWidgetItem(str(C['Orientation'])))
            table.setItem(k, 6, QTableWidgetItem(str(C['solidity'])))
            table.setItem(k, 7, QTableWidgetItem(str(C['rect_x'])))
            table.setItem(k, 8, QTableWidgetItem(str(C['rect_y'])))
            table.setItem(k, 9, QTableWidgetItem(str(C['rect_w'])))
            table.setItem(k, 10, QTableWidgetItem(str(C['rect_h'])))
            table.setItem(k, 11, QTableWidgetItem(str(C['L'])))
            table.setItem(k, 12, QTableWidgetItem(str(C['R'])))
            table.setItem(k, 13, QTableWidgetItem(str(C['T'])))
            table.setItem(k, 14, QTableWidgetItem(str(C['B'])))

    def highlight_cars(self):
        row_num = self.sender().currentRow()
        frame_color, mask, bw = self.frame_color.copy(), self.mask.copy(), self.bw.copy()
        mask = cv2.merge([mask, mask, mask])
        bw = cv2.merge([bw, bw, bw])
        cv2.drawContours(frame_color, self.CTF.cnts, row_num, (100, 100, 255), -1)
        cv2.drawContours(mask, self.CTF.cnts, row_num, (100, 100, 255), -1)
        cv2.drawContours(bw, self.CTF.cnts, row_num, (100, 100, 255), -1)

        pixmap_color = self.cv2_pic_to_pixmap(frame_color, 'bgr')  # 如果做切片，这句将无法调用
        pixmap_bw = self.cv2_pic_to_pixmap(bw, 'bgr')
        pixmap_mask = self.cv2_pic_to_pixmap(mask, 'bgr')
        self.label_frameBox.setPixmap(pixmap_color)
        self.label2.setPixmap(pixmap_bw)
        self.label3.setPixmap(pixmap_mask)

    def open_thresholder(self):
        thresholder = ColorThresholder.ColorThresholder()
        thresholder.color = self.frame_adjusted
        thresholder.show_mask()
        f = thresholder.exec_()
        if f:
            self.CTF.h_min = thresholder.horizontalSlider_HL.value()
            self.CTF.h_max = thresholder.horizontalSlider_HH.value()
            self.CTF.s_min = thresholder.horizontalSlider_SL.value()
            self.CTF.s_max = thresholder.horizontalSlider_SH.value()
            self.CTF.v_min = thresholder.horizontalSlider_VL.value()
            self.CTF.v_max = thresholder.horizontalSlider_VH.value()

    def openfilter(self):
        cntfilter = ContourFilter.ContourFilter()
        cntfilter.bw = self.bw
        cntfilter.show_filter()
        f = cntfilter.exec_()
        if f:
            print(22222)

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
                bgr = cv2.warpAffine(img, m, (w, h))
                show = self.cv2_pic_to_pixmap(bgr, 'bgr', self.label_frameBox)
                self.label_frameBox.setPixmap(show)

    def select_view(self):
        if self.action_viewCap.isChecked():
            self.x1, self.x2, self.y1, self.y2 = None, None, None, None
            self.statusBar.showMessage('请选择视区', 0)
            self.setCursor(Qt.CrossCursor)
            # cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), (255, 100, 0), 5)
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
                img = img[self.y1:self.y2, self.x1:self.x2]
                self.show = self.cv2_pic_to_pixmap(img, 'bgr')
                self.label_frameBox.setPixmap(self.show)
                return self.y1, self.y2, self.x1, self.x2

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

    def cv2_pic_to_pixmap(self, img, ptype, label=None):
        # 本函数可将cv2格式的图片转换为QPixmap能作用的格式，并设置图片尺寸为label控件的大小
        if ptype == 'rgb':
            tul = 3 * img.shape[1]  # 这个地方很奇怪，至今不能理解
            pic_q_image = QImage(img.data, img.shape[1], img.shape[0], tul, QImage.Format_RGB888)
        elif ptype == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tul = 3 * img.shape[1]  # 这个地方很奇怪，至今不能理解
            pic_q_image = QImage(img.data, img.shape[1], img.shape[0], tul, QImage.Format_RGB888)
        elif ptype == 'gray':
            tul = 1 * img.shape[1]  # 这个地方很奇怪，至今不能理解
            pic_q_image = QImage(img.data, img.shape[1], img.shape[0], tul, QImage.Format_Grayscale8)
        else:
            raise TypeError
        if label is None:
            pic_pixmap = QPixmap.fromImage(pic_q_image)
        else:
            pic_pixmap = QPixmap.fromImage(pic_q_image).scaled(label.width(), label.height())
        return pic_pixmap

    # def resizeEvent(self, event):
    #     self.label_frameBox.setGeometry(QtCore.QRect(0, 0, event.size().width(), event.size().height()-20))
    #     pic = QPixmap('E:\\have\Wall-E.jpg').scaled(self.label_frameBox.width(), self.label_frameBox.height())
    #     self.label_frameBox.setPixmap(pic)
    #     self.update()


class Thread(QThread):
    send_pic = pyqtSignal(np.ndarray, float, float)
    send_time = pyqtSignal(int)
    pp = True
    m_lock = QMutex()  # 互斥锁

    def run(self):
        self.cap = cv2.VideoCapture()
        self.cap.open(video_path)
        print(video_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)  # 帧率
        frame_num = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获得视频总帧数
        frame_time = int(frame_num/fps*1000)  # 视频总毫秒数
        self.send_time.emit(frame_time)  # 将视频总毫秒数传回
        while self.cap.isOpened():
            self.m_lock.lock()
            ret, frame = self.cap.read()
            if ret:
                # frame_cv2_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cur_num = self.cap.get(cv2.CAP_PROP_POS_FRAMES)  # 获得当前帧数
                cur_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)  # 当前毫秒数
                self.send_pic.emit(frame, cur_ms, cur_num)  # 将当前帧的画面和帧数传回
                print(cur_num)
                time.sleep(0.05)  # 控制视频播放的速度,计算量小时似乎不需要
            else:
                break
            self.m_lock.unlock()
        self.cap.release()

    def play_pause(self, f):
        print('pause:', f)
        print('isRunning?:', self.isRunning())
        if f:
            self.m_lock.lock()
        else:
            self.m_lock.unlock()

    def change_picShow(self, picshow):
        print(picshow)
        print(type(picshow))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, picshow)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    car_main = PlayerBox()
    car_main.showMaximized()
    app.exec_()
