# -- coding:utf-8 --
# author: ZQF time:2019/3/7

import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
if __name__ == '__main__':
    import ContourFilterUI
else:
    import widgets.ContourFilterUI as ContourFilterUI


class ContourFilter(QtWidgets.QDialog, ContourFilterUI.Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.bw, self.mask = None, None
        self.box = [[self.comboBox_11, self.doubleSpinBox_L11, self.doubleSpinBox_H11],
                    [self.comboBox_12, self.doubleSpinBox_L12, self.doubleSpinBox_H12],
                    [self.comboBox_13, self.doubleSpinBox_L13, self.doubleSpinBox_H13],
                    [self.comboBox_21, self.doubleSpinBox_L21, self.doubleSpinBox_H21],
                    [self.comboBox_22, self.doubleSpinBox_L22, self.doubleSpinBox_H22],
                    [self.comboBox_23, self.doubleSpinBox_L23, self.doubleSpinBox_H23],
                    [self.comboBox_31, self.doubleSpinBox_L31, self.doubleSpinBox_H31],
                    [self.comboBox_32, self.doubleSpinBox_L32, self.doubleSpinBox_H32],
                    [self.comboBox_33, self.doubleSpinBox_L33, self.doubleSpinBox_H33],
                    [self.comboBox_41, self.doubleSpinBox_L41, self.doubleSpinBox_H41],
                    [self.comboBox_42, self.doubleSpinBox_L42, self.doubleSpinBox_H42],
                    [self.comboBox_43, self.doubleSpinBox_L43, self.doubleSpinBox_H43],
                    [self.comboBox_51, self.doubleSpinBox_L51, self.doubleSpinBox_H51],
                    [self.comboBox_52, self.doubleSpinBox_L52, self.doubleSpinBox_H52],
                    [self.comboBox_53, self.doubleSpinBox_L53, self.doubleSpinBox_H53]]
        self.tableWidget.itemClicked.connect(self.highlight_cars)
        for each in self.box:
            each[0].currentIndexChanged.connect(self.show_filter)
            each[1].valueChanged.connect(self.show_filter)
            each[2].valueChanged.connect(self.show_filter)
        self.checkBox_hole.stateChanged.connect(self.show_filter)
        self.checkBox_lane.stateChanged.connect(self.show_filter)
        self.cnts2 = None
        # self.show()  # 要不要这句不影响？加上这句过后dialog变成了非模态的
        # self.exec_()  # 如果只用这句来执行程序，只有当程序退出后才会执行后面的语句
        self.show_filter()  # show update paint

    def show_filter(self):
        if self.bw is None:  # bw是传入待处理的图像
            pass
        else:
            self.mask = self.cnt_filter(self.bw.copy())  # 此处必须copy

            self.cv2_pic_to_pixmap(self.bw, 'gray', self.label)
            self.cv2_pic_to_pixmap(self.mask, 'gray', self.label_2)

    def cnt_filter(self, mask):
        if self.checkBox_hole.isChecked():  # 是否填充孔洞
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
                else:
                    if self.checkBox_lane.isChecked():  # 是否擦除车道线
                        mask = self.erase_lane(label, mask)  # 对于没有被过滤掉的对象擦除左右两端疑是车道线的东西
        # 在右侧显示过滤完后剩余轮廓的属性
        self.cnts2 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        Cs = []
        for cnt2 in self.cnts2:
            C2 = self.calc_characters(cnt2)
            Cs.append(C2)
        self.show_table(self.tableWidget, Cs)
        return mask

    def filter_or_not(self, C):
        for box in self.box:
            if box[0].currentText() == 'None':
                pass
            else:
                if box[1].value() <= C[box[0].currentText()] < box[2].value():
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
        mask, bw = self.mask.copy(), self.bw.copy()
        mask = cv2.merge([mask, mask, mask])
        bw = cv2.merge([bw, bw, bw])
        cv2.drawContours(mask, self.cnts2, row_num, (100, 100, 255), -1)
        cv2.drawContours(bw, self.cnts2, row_num, (100, 100, 255), -1)

        self.cv2_pic_to_pixmap(bw, 'bgr', self.label)
        self.cv2_pic_to_pixmap(mask, 'bgr', self.label_2)

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
            return pic_pixmap
        else:
            width, height = img.shape[1], img.shape[0]
            ratio = max(width/label.width(), height/label.height())
            pic_pixmap = QPixmap.fromImage(pic_q_image).scaled(width/ratio-2, height/ratio-2)
            label.setPixmap(pic_pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = ContourFilter()
    a = dialog.exec_()
    print(a)
    sys.exit()
