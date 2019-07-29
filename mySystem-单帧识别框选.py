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

from playerBoxUI import Ui_MainWindow
import ColorThresholder, ContourFilter


class ColorThreshFilter:
    def __init__(self):
        self.cars = {}
        self.Lanes, bkg, bk_cnts = self.bkg_process()
        self.cnts = None
        self.h_min, self.h_max = 0.388 * 180, 0.021 * 180  # 70, 4
        self.s_min, self.s_max = 0.000 * 256, 0.116 * 256  # 0, 30
        self.v_min, self.v_max = 0.398 * 256, 0.776 * 256  # 102, 199
        self.box = [['area', 30, 99999], ['Eccentricity', 0, 99999], ['Orientation', -90, 99999],
                    ['rect_w', 0, 99999], ['rect_h', 9, 99999], ['EquivDiameter', 0, 99999],
                    ['Extent', 0, 99999], ['solidity', 0, 99999], ['None', 0, 99999],
                    ['None', 0, 99999], ['None', 0, 99999], ['None', 0, 99999],
                    ['None', 0, 99999], ['None', 0, 99999], ['None', 0, 99999]]  # 提供最多同时15个过滤条件

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

    def do_morph(self, pic):
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        pic = cv2.dilate(pic, kernel1, iterations=1)
        pic = cv2.erode(pic, kernel2, iterations=1)
        # pic = cv2.morphologyEx(pic, cv2.MORPH_OPEN, kernel2, iterations=1)
        # pic = cv2.erode(pic, kernel)
        return pic

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

    def cnt_filter(self, contours, Lanes, mask, frame_color):
        for cnt in contours:
            C = self.calc_characters(cnt)
            if self.make_judgement(C):  # 返回值为1则过滤
                cv2.drawContours(mask, [cnt], 0, (0, 0, 0), -1)
            else:
                cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)
                mask, frame_color = self.erase_lane(cnt, C, mask, frame_color)
                pass
        return mask, frame_color

    def cnt_filter2(self, num, labels, mask, frame_color):
        for i in range(num):
            if i > 0:  # i=0是背景
                label = np.uint8(labels == i)
                _, cnts, hierarchy = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                C = self.calc_characters(cnts[0])
                if self.make_judgement(C):  # 返回值为1则过滤
                    cv2.drawContours(mask, cnts, 0, (0, 0, 0), -1)
                else:  # 对于没有被过滤掉的对象擦除左右两端疑是车道线的东西
                    mask, frame_color = self.erase_lane2(label, mask, frame_color)
        return mask, frame_color

    def make_judgement(self, C):
        for box in self.box:
            if box[0] == 'None':
                pass
            else:
                if box[1] <= C[box[0]] < box[2]:
                    continue
                else:
                    return 1  # 过滤掉
        return 0  # 不过滤

    def erase_lane(self, cnt, C, mask, frame_color):
        h, w = mask.shape
        length = 0
        k1 = 0
        while length < 10:  # 这个大循环删除左边车道线
            i, j, length = 0, 0, 0
            c = C['L'][0] + k1
            while 1:  # 向下找像素点，直到像素值不为255，每找到一个length+1，最后获得r1
                r1 = C['L'][1] + i
                if r1 >= h or c >= w:  # 为了避免下面mask索引的时候超出范围，一旦这里发现超限则停止循环
                    break
                value1 = mask[r1, c]
                if value1 == 255:
                    length += 1
                    i += 1
                else:
                    break
            while 1:  # 向上找像素点，直到像素值不为255，每找到一个length+1，最后获得r2
                r2 = C['L'][1] - j
                if r2 < 0 or c >= w:
                    break
                value2 = mask[r2, c]  # 注意中间那条线会统计两次，使length比实际多1
                if value2 == 255:
                    length += 1
                    j += 1
                else:
                    break
            if c >= w or c > C['R'][0]:
                break
            if length >= 10:
                break
            mask[max(r2, 0):min(r1, h-1), min(c, w-1)] = 0
            frame_color[max(r2, 0):min(r1, h-1), min(c, w-1), 2] = 255
            k1 += 1

        length = 0
        k2 = 0
        while length < 10:  # 这个大循环删除右边车道线
            i, j, length = 0, 0, 0
            c = C['R'][0] - k2
            while 1:
                r1 = C['R'][1] + i
                if r1 >= h or c < 0:
                    break
                value1 = mask[r1, c]
                if value1 == 255:
                    length += 1
                    i += 1
                else:
                    break
            while 1:
                r2 = C['R'][1] - j
                if r2 < 0 or c < 0:
                    break
                value2 = mask[r2, c]  # 注意中间那条线会统计两次，使length比实际多1
                if value2 == 255:
                    length += 1
                    j += 1
                else:
                    break
            if length >= 10:
                break
            if c < 0 or c < C['L'][0]:
                break
            mask[max(r2, 0):min(r1, h - 1), max(c, 0)] = 0
            frame_color[max(r2, 0):min(r1, h - 1), max(c, 0), 0] = 255
            k2 += 1
        return mask, frame_color

    def erase_lane2(self, label, mask, frame_color):
        vstack = np.sum(label, 0) < 10  # 纵向堆积像素个数小于10的列为True
        found = False
        for i in range(len(vstack)):
            if vstack[i] == False:
                a1 = i
                found = True
                break
        for j in range(len(vstack)):
            if vstack[-(j+1)] == False:
                a2 = -(j+1)
                found = True
                break
        if found:
            vstack[a1:a2] = False  # 一个对象中间部位如果存在堆积小于10的地方，同样不进行去除
        _mask = np.ones_like(mask)
        _mask[:] = vstack  # 锁定mask中纵向堆积像素个数小于10对应的整个列
        cnd = _mask & label  # 锁定目标中堆积像素个数小于10对应的所有像素点
        mask[cnd > 0] = 0  # 把锁定的像素点设置值为0
        # frame_color[:, :, 1][cnd > 0] = 255  # 在彩色图中将消除掉的位置标记为绿色
        return mask, frame_color

    def draw_and_text(self, cur, mask, frame_color):
        Cs = []
        _, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = cnts  # 弄成实例属性，方便在playerBox里面调用
        for cnt in cnts:
            C = self.calc_characters(cnt)
            Cs.append(C)
            # if C['area'] > 30:
            x, y, w, h = C['rect_x'], C['rect_y'], C['rect_w'], C['rect_h']
            if cur == 1:  # 如果是第一帧
                print(1, len(self.cars))
                self.cars[len(self.cars)] = (x, y, w, h)
                cv2.rectangle(frame_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.putText(frame_color, str(len(self.cars)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            else:  # 如果不是第一帧
                found = False
                if 5 < x < 830 and 5 < y < 120 :
                    for each in self.cars:  # 遍历储存的车辆，查找是否有与该轮廓邻近的信息，如果有，则认为该轮廓即搜索到的车辆
                        if -40 < x - self.cars[each][0] < 40 and -30 < y - self.cars[each][1] < 30:
                            self.cars[each] = (x, y, w, h)
                            # self.cars[each] = (cx, cy)
                            cv2.rectangle(frame_color, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 在frame上绘制矩形
                            # cv2.putText(frame_color, str(each), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                            found = True  # 如果找到了，设置一个flag
                    if not found:  # 如果没有找到，则把该轮廓标记为新的车辆
                    #     print(2, len(self.cars))
                    #     self.cars[len(self.cars)] = (x, y, w, h)
                        cv2.rectangle(frame_color, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 在frame上绘制矩形
                    #     cv2.putText(frame_color, str(len(self.cars)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            # else:
            #     cv2.drawContours(mask, [cnt], 0, (0, 0, 0), -1)
        return mask, frame_color, Cs

    def main_proceed(self, frame, cur_num):
        frame = self.adjust_frame(frame)  # 把视频调整为合适的大小 城市道路视频
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        adp_pic = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 6)

        frame_color = frame.copy()
        frame_adjusted = frame.copy()  # 备份一个，用来返回后传入ColorThresholder里面查看阈值设置的效果
        mask, pic = self.get_mask(frame)

        mask = cv2.bitwise_or(mask, adp_pic)  # 将局部阈值处理的结果和颜色过滤后的结果取了并集，目前看来效果不太理想
        bw = mask.copy()  # 复制一份，传回，用来在PlayerBox里面显示轮廓筛选之前的效果
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)  # 先把空洞填掉之后再找连通域
        num, labels = cv2.connectedComponents(mask)

        # mask, frame_color = self.cnt_filter(contours, self.Lanes, mask, frame_color)  # 从左右极点开始消除
        mask, frame_color = self.cnt_filter2(num, labels, mask, frame_color)  # 直接纵向堆叠消除
        # mask = self.do_morph(mask)
        mask, frame_color, mask_Cs = self.draw_and_text(cur_num, mask, frame_color)
        return frame_color, bw, mask, mask_Cs, frame_adjusted


class PlayerBox(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.connections()
        self.x1, self.x2, self.y1, self.y2 = None, None, None, None
        self.count = 1
        self.th = None
        self.frame_color, self.bw, self.mask = None, None, None
        self.frame_adjusted = None
        self.CTF = ColorThreshFilter()  # 实例化，调用专门处理车辆识别的类
        # self.pushButton_play_pause.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

    def connections(self):
        self.action_open.triggered.connect(self.open_video)
        self.action_save.triggered.connect(self.save_mask)
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
        video_path, video_type = QFileDialog.getOpenFileName(self, "选择视频", "..", "*.mov *.wmv *.mp4;;All Files(*)")
        if video_path == '':
            return
        else:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                self.show_frame(frame, 0, 0)
            self.th = Thread(self)
            self.th.send_pic.connect(self.show_frame)
            self.th.send_time.connect(self.set_time)
            self.horizontalSlider.sliderPressed.connect(self.set_button_false)
            self.horizontalSlider.sliderReleased.connect(lambda: self.change_frame(self.horizontalSlider.value()))
            self.pushButton_play_pause.clicked.connect(self.th.play_pause)
            self.th.start()

    def save_mask(self):
        fileName1, filetype = QFileDialog.getSaveFileName(self, "图片保存", ".", "Images (*.png *.bmp);;All Files (*)")
        fileName2, filetype = QFileDialog.getSaveFileName(self, "图片保存", ".", "Images (*.png *.bmp);;All Files (*)")
        if fileName1 == "":
            return
        else:
            try:
                cv2.imencode('.bmp', self.mask)[1].tofile(fileName1)
                cv2.imencode('.bmp', self.bw)[1].tofile(fileName2)
            except:
                QMessageBox.information(self, '提示', '尚未选择需要保存的图片', QMessageBox.Ok)

    def show_frame(self, frame_bgr, cur_ms, cur_num):  # 将传来的cv2格式图片转换为Qpixmap的并显示在label中
        # jpg的可以，png的不行
        self.frame_color, self.bw, self.mask, mask_Cs, self.frame_adjusted = self.CTF.main_proceed(frame_bgr, cur_num)
        self.show_table(self.tableWidget, mask_Cs)  # 显示右侧表格数据。Cs是储存了当前过滤完后一帧图像中所有轮廓所有特诊的一个列表

        # 显示这三张图片
        self.cv2_pic_to_pixmap(self.frame_color, 'bgr', self.label_frameBox)
        self.cv2_pic_to_pixmap(self.bw, 'gray', self.label2)
        self.cv2_pic_to_pixmap(self.mask, 'gray', self.label3)

        self.horizontalSlider.setValue(cur_num)  # 让slider滑块随视频播放而移动

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

        self.cv2_pic_to_pixmap(frame_color, 'bgr', self.label_frameBox)  # 如果做切片，这句将无法调用
        self.cv2_pic_to_pixmap(bw, 'bgr', self.label2)
        self.cv2_pic_to_pixmap(mask, 'bgr', self.label3)

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
        for i, each in enumerate(cntfilter.box):
            each[1].setValue(self.CTF.box[i][1])
            each[2].setValue(self.CTF.box[i][2])
        cntfilter.show_filter()
        f = cntfilter.exec_()
        if f:
            for i, each in enumerate(self.CTF.box):
                each[0] = cntfilter.box[i][0].currentText()
                each[1] = cntfilter.box[i][1].value()
                each[2] = cntfilter.box[i][2].value()

    def set_time(self, time, frame_num):
        print('total_frame:', time, frame_num)
        self.horizontalSlider.setMaximum(frame_num)  # 设置slider的最大值

    def set_button_false(self):
        self.th.pp = False
        self.pushButton_play_pause.setChecked(True)

    def change_frame(self, frame_num):
        print(frame_num)
        self.th.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.th.cap.read()
        self.show_frame(frame, 0, frame_num)

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
        # 本函数可将cv2格式的图片转换为QPixmap能作用的格式，并设置图片尺寸为适合label控件的大小
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
            return pic_pixmap
        else:
            width, height = img.shape[1], img.shape[0]
            ratio = max(width / label.width(), height / label.height())
            pic_pixmap = QPixmap.fromImage(pic_q_image).scaled(width / ratio-2, height / ratio-2)
            label.setPixmap(pic_pixmap)


class Thread(QThread):
    send_pic = pyqtSignal(np.ndarray, float, float)
    send_time = pyqtSignal(int, int)
    pp = False

    def run(self):
        self.cap = cv2.VideoCapture()
        self.cap.open(video_path)
        print(video_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)  # 帧率
        frame_num = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获得视频总帧数
        frame_time = int(frame_num/fps*1000)  # 视频总毫秒数
        self.send_time.emit(frame_time, frame_num)  # 将视频总毫秒数传回
        while self.cap.isOpened():
            if self.pp:
                ret, frame = self.cap.read()
                if ret:
                    cur_num = self.cap.get(cv2.CAP_PROP_POS_FRAMES)  # 获得当前帧数
                    cur_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)  # 当前毫秒数
                    self.send_pic.emit(frame, cur_ms, cur_num)  # 将当前帧的画面和帧数传回
                    time.sleep(0.15)  # 控制视频播放的速度,计算量小时似乎不需要
                else:
                    break
        self.cap.release()

    def play_pause(self):
        if not self.sender().isChecked():
            print('pause')
            self.pp = True
        else:
            print('start')
            self.pp = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    car_main = PlayerBox()
    car_main.showMaximized()
    app.exec_()
