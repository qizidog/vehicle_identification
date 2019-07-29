# -- coding:utf-8 --
# author: ZQF time:2018/12/15
'''
1.使用opencv的多目标追踪multitracker函数
2.可将追踪车辆的位置、速度信息以及视频路径、帧率等存入json文件
3.实现了计划中所有交通数据的计算和显示
'''

import sys
import os
# import time
import cv2
import numpy as np
import configparser
import json

# from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, \
     QTableWidgetItem, QInputDialog, QPushButton
# QDesktopWidget, QGraphicsScene, QDialog
from PyQt5.QtCore import QThread, pyqtSignal, Qt  # , pyqtSlot, QMutex
from PyQt5.QtGui import QPixmap, QImage

from playerBoxUI import Ui_MainWindow
import ColorThresholder, ContourFilter
from picbox import Mywin


class ColorThreshFilter:
    def __init__(self):
        # self.cars = {}
        self.mode = 0
        self.bboxes = []  # 目标追踪时存放信息的列表
        self.car_list = []  # 用来存放所有追踪到的车辆的编号
        self.multiTracker = cv2.MultiTracker_create()
        self.h_min, self.h_max = 0, 180  # hsv过滤的默认值
        self.s_min, self.s_max = 0, 255
        self.v_min, self.v_max = 0, 255
        self.reverse = True  # 在hsv过滤中是否反色
        self.auto_detect_flag = True  # 默认是开启自动检测模式的
        self.filter_pic = None
        self.filter_flag = True  # 控制是否执行轮廓过滤
        self.fill_hole_flag = True  # 控制是否填充孔洞（filter_flag为True时生效）
        self.erase_flag = True  # 控制是否执行消除车道线操作（filter_flag为True时生效）
        self.box = [['area', 200, 99999], ['Eccentricity', 0, 99999], ['Orientation', -90, 99999],
                    ['rect_w', 0, 99999], ['rect_h', 9, 99999], ['EquivDiameter', 0, 99999],
                    ['Extent', 0, 99999], ['solidity', 0, 99999], ['None', 0, 99999],
                    ['None', 0, 99999], ['None', 0, 99999], ['None', 0, 99999],
                    ['None', 0, 99999], ['None', 0, 99999], ['None', 0, 99999]]  # 提供最多同时15个过滤条件
        self.frames, self.fgbg, self.bkg = [], None, None  # 预定义背景差分法中使用的东西

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
        mask = (mask_h & mask_s & mask_v) * 255
        if self.reverse:
            mask = ~mask
        mask = np.uint8(mask)
        # masked_pic = cv2.bitwise_and(bgr, bgr, mask=mask)
        return mask

    def do_morph(self, pic):
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        # pic = cv2.dilate(pic, kernel1, iterations=1)
        pic = cv2.morphologyEx(pic, cv2.MORPH_CLOSE, kernel1, iterations=2)  # 闭运算
        pic = cv2.morphologyEx(pic, cv2.MORPH_OPEN, kernel1, iterations=1)  # 开运算
        # pic = cv2.erode(pic, kernel2, iterations=1)
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

    def cnt_filter(self, mask):
        if self.fill_hole_flag:  # 是否填充孔洞
            _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)  # 先把空洞填掉之后再找连通域
        num, labels = cv2.connectedComponents(mask)
        for i in range(num):
            if i > 0:  # i=0是背景
                label = np.uint8(labels == i)
                _, cnts, hierarchy = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                C = self.calc_characters(cnts[0])  # 计算这个连通域的轮廓特征
                if self.filter_or_not(C):  # 返回值为1则过滤
                    cv2.drawContours(mask, cnts, 0, (0, 0, 0), -1)
                elif self.erase_flag:  # 是否擦除车道线
                    mask = self.erase_lane(label, mask)  # 对于没有被过滤掉的对象擦除左右两端疑是车道线的东西
        return mask

    def filter_or_not(self, C):
        for box in self.box:
            if box[0] == 'None':
                pass
            else:
                if box[1] <= C[box[0]] < box[2]:
                    continue
                else:
                    return 1  # 过滤掉
        return 0  # 不过滤

    def erase_lane(self, label, mask):
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
        if np.sum(label)-np.sum(cnd) < 50:  # 把擦除后剩余面积小于50的去掉
            mask[(label-cnd) > 0] = 0
        # frame_color[:, :, 1][cnd > 0] = 255  # 在彩色图中将消除掉的位置标记为绿色
        return mask

    def draw_and_text(self, cur, mask, frame_color):
        Cs = []  # 里面存了一帧图里面所有轮廓目标的所有属性
        _, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 返回cnts，一帧里面的所有轮廓，方便在playerBox里面调用
        for cnt in cnts:
            C = self.calc_characters(cnt)
            Cs.append(C)
            # if C['area'] > 30:
            x, y, w, h = C['rect_x'], C['rect_y'], C['rect_w'], C['rect_h']
            if cur == 1:  # 如果是第一帧
                pass
                # print(1, len(self.cars))
                # self.cars[len(self.cars)] = (x, y, w, h)
                # cv2.rectangle(frame_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # cv2.putText(frame_color, str(len(self.cars)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            else:  # 如果不是第一帧
                found = False
                if 5 < x < 830 and 5 < y < 120:
                    pass
                    # for each in self.cars:  # 遍历储存的车辆，查找是否有与该轮廓邻近的信息，如果有，则认为该轮廓即搜索到的车辆
                    #     if -40 < x - self.cars[each][0] < 40 and -30 < y - self.cars[each][1] < 30:
                    #         self.cars[each] = (x, y, w, h)
                    #         # self.cars[each] = (cx, cy)
                    #         # cv2.rectangle(frame_color, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 在frame上绘制矩形
                    #         # cv2.putText(frame_color, str(each), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    #         found = True  # 如果找到了，设置一个flag
                    # if not found:  # 如果没有找到，则把该轮廓标记为新的车辆
                    #     print(2, len(self.cars))
                    #     self.cars[len(self.cars)] = (x, y, w, h)
                    #     cv2.rectangle(frame_color, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 在frame上绘制矩形
                    #     cv2.putText(frame_color, str(len(self.cars)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            # else:
            #     cv2.drawContours(mask, [cnt], 0, (0, 0, 0), -1)
        return frame_color

    def add_into_tracker(self, mask, frame_color, cur_num):
        if cur_num == 1:  # 这个帧数非常关键，如果没有识别，去查thread里面的初始帧是0还是1
            boxes = np.array([])
            car_list = np.array(self.car_list.copy())
            # num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            # for i in range(num):
            #     if i > 0:
            #         bbox = stats[i][:4]
            #         self.multiTracker.add(cv2.TrackerKCF_create(), frame_color, tuple(bbox))  # cv2.TrackerMOSSE_create()
        else:
            # 刷新刷新刷新
            ret, boxes = self.multiTracker.update(frame_color)  # 跟踪对象消失后boxes中的数据不会减少
            print('boxes', self.multiTracker.getObjects())
            # print('ret', ret)
            if isinstance(boxes, np.ndarray):
                scope_mask = (0 < boxes[:, 0]) & ((boxes[:, 0]+boxes[:, 2]) < frame_color.shape[1])
                boxes = boxes[scope_mask, :]  # 过滤掉从左右两边界离开的目标
                # print(scope_mask, boxes, self.car_list)
                car_list = np.array(self.car_list.copy())[scope_mask]
                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))  # 框的左上角x, y坐标
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))  # 框的右下角x, y坐标
                    cv2.rectangle(frame_color, p1, p2, (160, 80, 40), 2, 1)
                    cv2.putText(frame_color, str(car_list[i]), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
            else:
                boxes = np.array([])  # 做一个空的np数组以避免报错
                car_list = np.array(self.car_list.copy())
            # 自动检测flag，if true
            if self.auto_detect_flag:
                entrance = mask[:, 0:280].copy()
                num, labels, stats, centroids = cv2.connectedComponentsWithStats(entrance)
                # cv2.imshow('entrance', entrance)
                # cv2.waitKey(1)
                for i in range(num):
                    found = False
                    if i > 0 and stats[i][4] > 200 and stats[i][0] > 5:  # 如果识别到面积大于200且不沾边的对象
                    # if i > 0 and stats[i][0] > 5:  # 如果识别到不沾边的对象
                        for j in boxes:  # 遍历判断是否为已存在的车辆
                            if self.isSameCar(stats[i][0:4], j):
                                found = True
                                break
                        if not found:
                            bbox = stats[i][:4]
                            self.multiTracker.add(cv2.TrackerKCF_create(), frame_color, tuple(bbox))
                            self.car_list.append(len(self.car_list))
                            # self.bboxes.append(bbox)
        return frame_color, boxes, car_list

    def isSameCar(self, box1, box2):  # 根据两矩形框的相交面积判断是否为同一辆车
        min_x = max(box1[0], box2[0])
        min_y = max(box1[1], box2[1])
        max_x = min((box1[0]+box1[2]), (box2[0]+box2[2]))
        max_y = min((box1[1]+box1[3]), (box2[1]+box2[3]))
        if min_x > max_x or min_y > max_y:  # 此情况下两矩形无交集
            return 0  # 不是同一辆车
        else:
            s = (max_x-min_x)*(max_y-min_y)  # 两矩形相交的面积
            r = s / min(box1[2]*box1[3], box2[2]*box2[3])  # 相交面积与小面积的比值
            if r > 0.2:  # 比值大于一阈值
                return 1  # 判定为同一辆车
            else:
                return 0

    def main_proceed(self, frame, cur_num):
        frame_color = frame.copy()
        # 自动检测flag，if true
        if self.auto_detect_flag:
            if self.mode == 0:
                mask = self.feature_based(frame, cur_num)  # 使用基于特征的方法进行车辆检测
            elif self.mode == 1:
                mask = self.frame_diff(frame, cur_num)  # 使用帧差法进行车辆检测
            elif self.mode == 2:
                mask = self.MOG2(frame, cur_num)  # 使用混合高斯模型进行车辆检测
            else:
                print('没有这种检测模式！')
                raise AttributeError
            self.filter_pic = mask.copy()  # 供轮廓过滤控件显示用的图片
            if self.filter_flag:
                mask = self.cnt_filter(mask)  # 目标过滤，纵向堆叠消除
        else:
            mask = np.zeros_like(frame_color[:, :, 0])  # 必须是np.ndarray格式，否则会报错
            self.filter_pic = None  # 供轮廓过滤控件显示用的图片
        frame_color, boxes, car_list = self.add_into_tracker(mask, frame_color, cur_num)
        return frame_color, mask, boxes, car_list

    def feature_based(self, frame, cur_num):  # 颜色+形状过滤
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        adp_pic = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 6)
        mask = self.get_mask(frame)  # 通过hsv通道筛选
        mask = cv2.bitwise_or(mask, adp_pic)  # 将局部阈值处理的结果和颜色过滤后的结果取了并集，目前看来效果不太理想

        # mask = mask[:, 0:280].copy()  # 注释掉此行即可复原
        # mask = self.do_morph(mask)
        return mask

    def frame_diff(self, frame, cur_num):  # 帧差+形态学+形状过滤
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cur_num < 3:
            self.frames.append(gray)
            bw = cv2.absdiff(gray, gray)
        else:
            self.frames.append(gray)
            frame0 = self.frames.pop(0)
            bw = cv2.absdiff(frame0, gray)
            bw = cv2.threshold(bw, 13, 255, cv2.THRESH_BINARY)[1]
        mask = self.do_morph(bw)

        # _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)  # 先把空洞填掉之后再找连通域
        # num, labels = cv2.connectedComponents(mask)
        # mask = self.cnt_filter(mask)  # 目标过滤，纵向堆叠消除
        return mask

    def MOG2(self, frame, cur_num):  # 混合高斯背景减除+形状过滤
        if cur_num == 1:  # 如果没生效，去查thread里面的cur_num初始帧是0还是1
            self.fgbg = cv2.createBackgroundSubtractorMOG2()  # MOG2还可以换成KNN
            # self.fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
            # self.fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()  # 同问，此外最后的GMG还可以换成CNT/GSOC/LSBP/MOG
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.fgbg.apply(mask)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
        return mask

    def main_proceed4(self, frame, cur_num):  # hsv通道筛选+静态背景差分（hsv筛选后的）
        if cur_num == 0:
            self.bkg = cv2.imread('bkg_gap20_cur641.0.jpg')
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        bw = self.get_mask(frame)
        bkg = self.get_mask(self.bkg)
        cv2.imshow('bkg', bkg)
        print(bkg.shape)
        print(bw.shape)
        mask = cv2.absdiff(bw, bkg)
        return bw, mask


class VideoThread(QThread):
    send_pic = pyqtSignal(np.ndarray, np.ndarray, float)
    send_time = pyqtSignal(int, int)
    send_data = pyqtSignal(np.ndarray, np.ndarray, float)
    pp = False  # flag of play and pause

    def __init__(self):
        super().__init__()
        self.CF = ColorThreshFilter()  # 实例化，调用专门处理车辆识别的类
        self.theta = 0  # 默认的旋转角度
        self.upRow, self.downRow, self.leftCol, self.rightCol = 0, None, 0, None  # 默认截取的窗口大小
        # self.upRow, self.downRow, self.leftCol, self.rightCol = 560, 650, 470, 1030  # 默认截取的窗口大小
        self.cap = cv2.VideoCapture()  # 不要放到run里面，不然会导致未选择视频时变更检测模式报错
        self.frame = None  # 预定义调整大小后的图像，不要删！
        self.cur_num = 0  # 预定义当前的帧数

    def run(self):
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
                    cur_num = self.cap.get(cv2.CAP_PROP_POS_FRAMES)  # 获得当前帧数（获取等前帧在read之后，所以没有第0帧）
                    self.cur_num = cur_num
                    # cur_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)  # 当前毫秒数
                    self.frame = self.adjust_frame(frame)
                    frame_color, mask, boxes, car_list = self.CF.main_proceed(self.frame, cur_num)
                    # print('type boxes:', type(boxes))
                    self.send_data.emit(boxes, car_list, cur_num)  # 已经在add_into_tracker里面确保boxes是np.array类型
                    self.send_pic.emit(frame_color, mask, cur_num)  # 将当前帧的画面和帧数传回
                    # time.sleep(0.1)  # 控制视频播放的速度,计算量小时似乎不需要
                else:
                    break
        print('cap has been released')
        self.cap.release()

    def play_pause(self):
        if not self.sender().isChecked():
            print('pause')
            self.pp = True
        else:
            print('start')
            self.pp = False

    def adjust_frame(self, frame):  # 旋转和截取感兴趣区域
        rows, cols, ch = frame.shape
        if self.downRow is None:
            self.downRow = rows
        if self.rightCol is None:
            self.rightCol = cols
        M = cv2.getRotationMatrix2D((cols/2, rows/2), self.theta, 1)  # 三个参数分别是旋转中心，旋转角度，比例
        frame = cv2.warpAffine(frame, M, (cols, rows))
        frame = frame[self.upRow:self.downRow, self.leftCol:self.rightCol]
        return frame


class DataThread(QThread):
    send_micro_data = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    send_macro_data = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.json_path = None
        self.calib = 1  # 像素距离/实际距离
        self.last_pos = np.empty([1, 2])
        self.last_speed = np.empty([1, 1])
        self.match = None
        self.fps = None
        self.freq = 3

    def micro_process(self, boxes, car_list, cur_num):
        if len(boxes) > 0 and not cur_num % int(self.fps/self.freq):
            pos = boxes[:, 0:2] + boxes[:, 2:4]/2
            match = dict(zip(car_list, pos))  # 建立车辆编号和对应车辆位置信息的匹配关系
            print('connect successfully!')
            if len(pos) == len(self.last_pos):
                speed = np.round(self.calc_distance(pos-self.last_pos) / self.calib * 3, 2)  # 1/3s内产生的位移，所以需要乘以3
            elif len(pos) > len(self.last_pos):  # 如果车辆数增加了  # todo y坐标导致新增车辆速度偏大
                mov = pos.copy()
                for i, p in enumerate(self.last_pos):
                    mov[i] = pos[i] - p  # 第i辆车的位移
                speed = np.round(self.calc_distance(mov) / self.calib * self.freq, 2)
            else:  # 如果车辆数减少了
                mov = pos.copy()
                for i, (key, value) in enumerate(match.items()):
                    mov[i] = value - self.match[key]  # 第i辆车的位移
                speed = np.round(self.calc_distance(mov) / self.calib * self.freq, 2)
            self.last_pos = pos  # 历史位置更新
            self.last_speed = speed  # 历史速度更新
            self.match = match  # 匹配关系更新
            self.send_micro_data.emit(car_list, pos, speed)
            if self.json_path:
                self.dump_json(match.copy(), speed.copy(), cur_num)  # 把信息保存到磁盘文件里面

    def calc_distance(self, arr):
        return np.sqrt(np.square(arr[:, 0]) + np.square(arr[:, 1])).astype(int)

    def dump_json(self, match, speed, cur_num):
        new_match = {}  # 用来存新构建的指定格式的数据
        for i, (key, value) in enumerate(match.items()):
            v = value.tolist()
            v.append(speed[i])
            new_match[key] = v
        with open(self.json_path, 'r') as f:  # 先把json文件中的数据读出来
            try:
                load_data = json.load(f)
            except json.decoder.JSONDecodeError:  # 如果是个新的空文件
                load_data = {'data': {}, 'cur_num': '', 'fps': self.fps, 'frequency': self.freq, 'video_path': video_path}
        for i, (key, value) in enumerate(new_match.items()):  # 按照车辆的编号追加数据到读取的数据中
            if str(key) in load_data['data']:  # 如果字典中已经有这个键了
                load_data['data'][str(key)].append(value)
            else:  # 如果字典中没有这个键
                load_data['data'][str(key)] = []
                load_data['data'][str(key)].append(value)
        load_data['cur_num'] = cur_num
        with open(self.json_path, 'w') as f:  # 追加完数据后重新写入json文件
            json.dump(load_data, f, sort_keys=True, indent=4)

    def macro_process(self):  # cur_num：当前视频帧数
        with open(self.json_path, 'r') as f:
            try:
                load_data = json.load(f)
            except json.decoder.JSONDecodeError:  # 如果是个新的空文件
                print('该json文件暂时无数据')
                return
        speed_t, speed_s = self.get_speed(load_data)
        cur_num = int(load_data['cur_num'])
        fps = load_data['fps']
        Q = int(len(load_data['data']) * fps * 3600 / cur_num)  # 交通量：Q = car_num / (cur_num/fps/3600) 辆/h
        K = int(Q / speed_s)  # 辆/km
        h_s = round(1000 / K, 2)  # m/辆
        h_t = round(3600 / Q, 2)  # s/辆
        self.send_macro_data.emit([Q, speed_s, speed_t, K, h_s, h_t])

    def get_speed(self, load_data):
        speed = []  # 存储提取的地点车速
        value = list(load_data['data'].values())
        car_num = len(value)
        for i in range(car_num):
            try:
                speed.append(value[i][1][2])  # 以每个车辆提取到的第二个速度作为其地点车速
            except IndexError:
                car_num -= 1
        speed_t = np.mean(speed)*3.6  # 时间平均速度 km/h
        speed_s = car_num/np.sum(1/np.array(speed))*3.6  # 空间平均速度 km/h
        return np.round(speed_t, 2), np.round(speed_s, 2)


class PlayerBox(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.add_buttom()
        self.x1, self.x2, self.y1, self.y2 = None, None, None, None  # 供调整旋转角度时使用
        self.vth = VideoThread()  # 播放视频的线程
        self.dth = DataThread()  # 处理数据的线程
        self.connections()  # 将一些信号和槽关联起来

    def add_buttom(self):
        self.refresh_btn = QPushButton('获取数据')
        self.tableWidget_macro.setCellWidget(7, 2, self.refresh_btn)

    def connections(self):
        self.action_open.triggered.connect(self.open_video)
        self.action_save.triggered.connect(self.save_json)
        self.action_rotate.triggered.connect(self.rotate_frame)
        self.action_viewCap.triggered.connect(self.select_view)
        self.action_calib.triggered.connect(self.set_calib)
        self.action_drawbox.triggered.connect(self.select_box)
        self.action_ColorThreshholder.triggered.connect(self.open_thresholder)
        self.action_ContourFilter.triggered.connect(self.open_filter)
        self.action_readConfig.triggered.connect(self.read_config)
        self.action_saveConfig.triggered.connect(self.save_config)
        self.action_getbkg.triggered.connect(self.open_getbkg)
        self.action_ColorThreshholder.triggered.connect(self.select_detect_mode)
        self.action_frameDiff.triggered.connect(self.select_detect_mode)
        self.action_MOG2.triggered.connect(self.select_detect_mode)
        self.action_autoDetect.triggered.connect(self.control_detect_model)
        self.vth.send_pic.connect(self.show_frame)
        self.vth.send_time.connect(self.set_time)
        self.vth.send_data.connect(self.dth.micro_process)
        self.dth.send_micro_data.connect(self.show_micro_data)
        self.dth.send_macro_data.connect(self.show_macro_data)
        # self.refresh_btn.clicked.connect(self.test)

    def test(self):
        a = self.vth.CF.multiTracker.trackerList()
        print('a', a, type(a))

    def open_video(self):
        global video_path
        video_path, video_type = QFileDialog.getOpenFileName(self, "选择视频", "..", "*.mov *.wmv *.mp4;;All Files(*)")
        if video_path == '':
            return
        else:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            self.dth.fps = int(cap.get(cv2.CAP_PROP_FPS))  # 把帧率信息传递给数据处理的线程
            cap.release()
            if ret:
                self.auto_config()  # 自动读取配置文件
                frame = self.vth.adjust_frame(frame)  # 裁剪一下图片大小
                self.show_frame(frame, None, 0)  # 打开视频后自动显示第一帧图片
                self.action_rotate.setEnabled(True)  # 解除三个按钮的锁定
                self.action_viewCap.setEnabled(True)
                self.action_drawbox.setEnabled(True)
                self.action_calib.setEnabled(True)
                self.pushButton_play_pause.clicked.connect(self.vth.play_pause)
                self.horizontalSlider.sliderPressed.connect(self.set_button_false)
                self.horizontalSlider.sliderReleased.connect(lambda: self.change_frame(self.horizontalSlider.value()))
                self.vth.start()

    def auto_config(self):
        config_path = str(os.path.split(video_path)[1].split('.')[0]) + '.ini'  # 配置文件的名字
        if os.path.exists(config_path):  # 如果存在和视频名字相同的配置文件
            cf = configparser.ConfigParser()
            cf.read(config_path)  # 读取配置文件
            self.vth.upRow = cf.getint('view', 'up')
            self.vth.downRow = cf.getint('view', 'down')
            self.vth.leftCol = cf.getint('view', 'left')
            self.vth.rightCol = cf.getint('view', 'right')
            self.vth.theta = cf.getfloat('rotate', 'theta')
            self.vth.CF.h_min = cf.getint('hsv', 'hl')
            self.vth.CF.h_max = cf.getint('hsv', 'hh')
            self.vth.CF.s_min = cf.getint('hsv', 'sl')
            self.vth.CF.s_max = cf.getint('hsv', 'sh')
            self.vth.CF.v_min = cf.getint('hsv', 'vl')
            self.vth.CF.v_max = cf.getint('hsv', 'vh')
            self.vth.CF.reverse = cf.getboolean('hsv', 'reverse')

    def save_json(self):
        json_path, filetype = QFileDialog.getOpenFileName(self, "选择数据保存路径", "./car_datas", "Json (*.json);;All Files (*)")
        if json_path == "":
            return
        else:
            self.dth.json_path = json_path
            self.refresh_btn.clicked.connect(self.dth.macro_process)

    def show_frame(self, frame_bgr, mask, cur_num):  # 将传来的cv2格式图片转换为Qpixmap的并显示在label中
        # 显示这三张图片
        # self.cv2_pic_to_pixmap(self.mask, 'gray', self.label3)
        self.cv2_pic_to_pixmap(frame_bgr, 'bgr', self.label_frameBox)  # 显示追踪效果
        self.cv2_pic_to_pixmap(mask, 'gray', self.label2)  # 显示检测效果
        self.horizontalSlider.setValue(cur_num)  # 让slider滑块随视频播放而移动
        print('cur:', cur_num)  # 显示当前帧数

    def change_frame(self, frame_num):
        print('jump to frame:', frame_num)
        self.vth.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.vth.cap.read()
        frame = self.vth.adjust_frame(frame)
        frame_color, mask = self.vth.CF.main_proceed(frame, frame_num)[0:2]
        self.show_frame(frame_color, mask, frame_num)

    def show_micro_data(self, car_list, pos, speed):
        # 后面设定为根据sender()的不同自动选择显示的表格
        self.tableWidget_micro.setRowCount(len(car_list))
        for i, each_car in enumerate(car_list):
            self.tableWidget_micro.setItem(i, 0, QTableWidgetItem(str(each_car)))
        for i, each_pos in enumerate(pos):
            self.tableWidget_micro.setItem(i, 1, QTableWidgetItem(str(each_pos)))
        for j, each_speed in enumerate(speed):
            self.tableWidget_micro.setItem(j, 2, QTableWidgetItem(str(each_speed)))
        item = QTableWidgetItem(str(len(self.vth.CF.car_list))+' 辆')
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置水平和垂直方向居中对齐
        self.tableWidget_macro.setItem(0, 1, item)

    def show_macro_data(self, macro_data):
        Q, speed_s, speed_t, K, h_s, h_t = macro_data
        items = [QTableWidgetItem(str(Q)+' 辆/h'), QTableWidgetItem(str(speed_s)+' km/h'),
                QTableWidgetItem(str(speed_t)+' km/h'), QTableWidgetItem(str(K)+' 辆/km'),
                QTableWidgetItem(str(h_s)+' m/辆'), QTableWidgetItem(str(h_t)+' s/辆')]
        for i, item in enumerate(items):
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.tableWidget_macro.setItem(i+1, 1, item)

    def control_detect_model(self):
        self.vth.CF.auto_detect_flag = self.action_autoDetect.isChecked()

    def select_detect_mode(self):  # 选择使用车辆检测的方式
        modes = [self.action_ColorThreshholder, self.action_frameDiff, self.action_MOG2]
        self.set_button_false()
        self.vth.CF.frames = []  # 把帧差法用来存背景的列表清空
        self.vth.CF.multiTracker = cv2.MultiTracker_create()  # 追踪器重新创建
        self.vth.CF.car_list = []  # 把存放所有追踪到的车辆编号的列表清空
        self.vth.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 视频重头开始播放(小心thread里面改变cap定义的位置可能导致这里报错)
        for i, each_mode in enumerate(modes):
            if each_mode == self.sender():
                each_mode.setChecked(True)
                self.vth.CF.mode = i
            else:
                each_mode.setChecked(False)

    def open_thresholder(self):
        self.set_button_false()
        thresholder = ColorThresholder.ColorThresholder()
        if self.vth.frame is not None:  # 避免在视频播放之前打开hsv控件报错
            thresholder.color = cv2.GaussianBlur(self.vth.frame, (3, 3), 0)
        thresholder.horizontalSlider_HL.setValue(self.vth.CF.h_min)
        thresholder.horizontalSlider_HH.setValue(self.vth.CF.h_max)
        thresholder.horizontalSlider_SL.setValue(self.vth.CF.s_min)
        thresholder.horizontalSlider_SH.setValue(self.vth.CF.s_max)
        thresholder.horizontalSlider_VL.setValue(self.vth.CF.v_min)
        thresholder.horizontalSlider_VH.setValue(self.vth.CF.v_max)
        thresholder.checkBox.setChecked(self.vth.CF.reverse)
        # thresholder.show_mask()
        f = thresholder.exec_()
        if f:
            self.vth.CF.h_min = thresholder.horizontalSlider_HL.value()
            self.vth.CF.h_max = thresholder.horizontalSlider_HH.value()
            self.vth.CF.s_min = thresholder.horizontalSlider_SL.value()
            self.vth.CF.s_max = thresholder.horizontalSlider_SH.value()
            self.vth.CF.v_min = thresholder.horizontalSlider_VL.value()
            self.vth.CF.v_max = thresholder.horizontalSlider_VH.value()
            self.vth.CF.reverse = thresholder.checkBox.isChecked()

    def open_filter(self):
        self.set_button_false()  # 暂停播放
        cntfilter = ContourFilter.ContourFilter()

        # 值传递
        cntfilter.bw = self.vth.CF.filter_pic  # 把需要显示的当前帧传递给过滤控件
        cntfilter.checkBox_hole.setChecked(self.vth.CF.fill_hole_flag)
        cntfilter.checkBox_lane.setChecked(self.vth.CF.erase_flag)
        for i, each in enumerate(cntfilter.box):
            each[1].setValue(self.vth.CF.box[i][1])
            each[2].setValue(self.vth.CF.box[i][2])
        # cntfilter.show_filter()
        f = cntfilter.exec_()
        if f:
            self.vth.CF.fill_hole_flag = cntfilter.checkBox_hole.isChecked()
            self.vth.CF.erase_flag = cntfilter.checkBox_lane.isChecked()
            for i, each in enumerate(self.vth.CF.box):
                each[0] = cntfilter.box[i][0].currentText()
                each[1] = cntfilter.box[i][1].value()
                each[2] = cntfilter.box[i][2].value()

    def open_getbkg(self):
        picbox = Mywin()
        picbox.show()

    def set_time(self, time, frame_num):
        print('total_time:', time, 'total_frame:', frame_num)
        self.horizontalSlider.setMaximum(frame_num)  # 设置slider的最大值

    def set_button_false(self):
        self.vth.pp = False
        self.pushButton_play_pause.setChecked(True)

    def get_points(self, event, x, y, flags, frame):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x1, self.y1 = x, y
            print('起点坐标：', x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.x2, self.y2 = x, y
            print('终点坐标：', x, y)
            cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255), 2)

    def rotate_frame(self):
        self.set_button_false()
        cv2.namedWindow('rotate frame', cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        frame2 = frame.copy()
        cv2.namedWindow('rotate frame', 0)
        cv2.setMouseCallback('rotate frame', self.get_points, frame)  # 输出鼠标点击位置的坐标
        while 1:
            cv2.imshow('rotate frame', frame)
            k = cv2.waitKey(100)
            if k == ord('c'):  # 重置图片
                frame = frame2.copy()
                cv2.setMouseCallback('rotate frame', self.get_points, frame)  # 重新建立连接
            elif k == ord('y'):  # 确认并退出
                try:
                    theta = cv2.fastAtan2((self.y2 - self.y1), (self.x2 - self.x1))  # 把需要旋转的角度计算出来
                except ZeroDivisionError:
                    print('zero')
                    theta = 90  # 以度为单位
                self.vth.theta = theta
                print(theta)  # 赋值
                cv2.destroyWindow('rotate frame')
                frame2 = self.vth.adjust_frame(frame2)
                self.show_frame(frame2, None, 0)
                break
            elif k == 27:  # 退出不做修改
                cv2.destroyAllWindows()
                break

    def select_view(self):
        self.set_button_false()
        cv2.namedWindow('selectROI', cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        frame2 = frame.copy()
        rows, cols, ch = frame.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), self.vth.theta, 1)  # 三个参数分别是旋转中心，旋转角度，比例
        frame = cv2.warpAffine(frame, M, (cols, rows))

        bbox = list(cv2.selectROI('selectROI', frame))
        cv2.destroyWindow('selectROI')
        if bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0:
            pass  # if canceled the select, do nothing
        else:
            self.vth.leftCol, self.vth.rightCol, self.vth.upRow, self.vth.downRow = bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]
        print(self.vth.upRow, self.vth.downRow, self.vth.leftCol, self.vth.rightCol)
        # self.CF.frames = []  # 似乎是在使用帧差法的时候需要这句话，不管了，先注释掉
        frame2 = self.vth.adjust_frame(frame2)
        self.show_frame(frame2, None, 0)

    def set_calib(self):
        self.set_button_false()  # 把视频暂停了
        if self.vth.frame is None:
            cap = cv2.VideoCapture(video_path)
            frame = self.vth.adjust_frame(cap.read()[1])
            cap.release()
        else:
            frame = self.vth.frame.copy()
        frame2 = frame.copy()
        cv2.namedWindow('calibration', 0)
        cv2.setMouseCallback('calibration', self.get_points, frame)  # 输出鼠标点击位置的坐标
        while 1:
            cv2.imshow('calibration', frame)
            k = cv2.waitKey(100)
            if k == ord('c'):  # 重置图片
                frame = frame2.copy()
                cv2.setMouseCallback('calibration', self.get_points, frame)  # 重新建立连接
            elif k == ord('y'):  # 确认并退出
                d_pixel = ((self.y2 - self.y1)**2 + (self.x2 - self.x1)**2)**0.5
                d_real, ok = QInputDialog.getDouble(self, '像素标定', '所绘线段的实际长度(m)：', min=0.01, decimals=2)
                if ok:
                    self.dth.calib = d_pixel/d_real
                    print('像素距离：', d_pixel, '实际距离', d_real, '像素/实际', self.dth.calib)
                cv2.destroyWindow('calibration')
                break
            elif k == 27:  # 退出不做修改
                cv2.destroyWindow('calibration')
                break

    def select_box(self):  # 手动绘制待跟踪目标
        self.set_button_false()  # 把视频暂停了
        if self.vth.frame is None:
            cap = cv2.VideoCapture(video_path)
            frame = self.vth.adjust_frame(cap.read()[1])
            cap.release()
        else:
            frame = self.vth.frame.copy()

        cv2.namedWindow('MultiTracker', cv2.WINDOW_NORMAL)
        print('选择一个目标，按下space或enter键后确认选择，继续选择下一个目标，全部选择结束后按Esc键退出')
        bboxes = cv2.selectROIs("MultiTracker", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow('MultiTracker')
        for bbox in bboxes:
            self.vth.CF.multiTracker.add(cv2.TrackerKCF_create(), frame, tuple(bbox))
            self.vth.CF.car_list.append(len(self.vth.CF.car_list))

    def read_config(self):
        config_path, config_type = QFileDialog.getOpenFileName(self, "选择配置文件", ".", "*.ini *.conf;;All Files(*)")
        if config_path == '':
            return
        else:
            cf = configparser.ConfigParser()
            cf.read(config_path)  # 读取配置文件
            self.vth.upRow = cf.getint('view', 'up')
            self.vth.downRow = cf.getint('view', 'down')
            self.vth.leftCol = cf.getint('view', 'left')
            self.vth.rightCol = cf.getint('view', 'right')
            self.vth.theta = cf.getfloat('rotate', 'theta')
            self.vth.CF.h_min = cf.getint('hsv', 'hl')
            self.vth.CF.h_max = cf.getint('hsv', 'hh')
            self.vth.CF.s_min = cf.getint('hsv', 'sl')
            self.vth.CF.s_max = cf.getint('hsv', 'sh')
            self.vth.CF.v_min = cf.getint('hsv', 'vl')
            self.vth.CF.v_max = cf.getint('hsv', 'vh')
            self.vth.CF.reverse = cf.getboolean('hsv', 'reverse')

    def save_config(self):
        config_path = str(os.path.split(video_path)[1].split('.')[0]) + '.ini'
        config_path, config_type = QFileDialog.getSaveFileName(self, "保存配置文件", config_path, "*.ini;;*.conf;;All Files(*)")
        if config_path == "":
            return
        else:
            cf = configparser.ConfigParser()
            cf.add_section("rotate")
            cf.set("rotate", "theta", str(self.vth.theta))
            cf.add_section("view")
            cf.set("view", "up", str(self.vth.upRow))
            cf.set("view", "down", str(self.vth.downRow))
            cf.set("view", "left", str(self.vth.leftCol))
            cf.set("view", "right", str(self.vth.rightCol))
            cf.add_section("hsv")
            cf.set("hsv", "hl", str(self.vth.CF.h_min))
            cf.set("hsv", "hh", str(self.vth.CF.h_max))
            cf.set("hsv", "sl", str(self.vth.CF.s_min))
            cf.set("hsv", "sh", str(self.vth.CF.s_max))
            cf.set("hsv", "vl", str(self.vth.CF.v_min))
            cf.set("hsv", "vh", str(self.vth.CF.v_max))
            cf.set("hsv", "reverse", str(self.vth.CF.reverse))
            with open(config_path, "w+") as f:
                cf.write(f)

    def cv2_pic_to_pixmap(self, img, ptype, label=None):
        # 本函数可将cv2格式的图片转换为QPixmap能作用的格式，并设置图片尺寸为适合label控件的大小
        if img is None:  # 如果没有传图片进来就什么都不干
            return 0
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = PlayerBox()
    main_window.show()
    # main_window.showMaximized()
    app.exec_()
