# -- coding: utf-8 --
# author: ZQF  time: 2019/5/18 22:54

import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QLabel
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QImage
from functools import partial
if __name__ == '__main__':
    import MorphAdjusterUI
else:
    import widgets.MorphAdjusterUI as MorphAdjusterUI


class MorphAdjuster(QtWidgets.QDialog, MorphAdjusterUI.Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.before, self.after = None, None
        self.methods = {'矩形': cv2.MORPH_RECT, '椭圆形': cv2.MORPH_ELLIPSE, '十字形': cv2.MORPH_CROSS,
                        '开运算': cv2.MORPH_OPEN, '闭运算': cv2.MORPH_CLOSE,
                        '膨胀': cv2.MORPH_DILATE, '腐蚀': cv2.MORPH_ERODE}
        self.comboboxes = [self.comboBox_kernel1, self.comboBox_kernel2, self.comboBox_morph1, self.comboBox_morph2]
        self.spinboxes = [self.spinBox_x1, self.spinBox_y1, self.spinBox_circ1,
                          self.spinBox_x2, self.spinBox_y2, self.spinBox_circ2]
        self.morph1, self.morph2 = None, None
        for each in self.comboboxes:
            each.currentIndexChanged.connect(self.show_morph)
        for each in self.spinboxes:
            each.valueChanged.connect(self.show_morph)

    def show_morph(self):
        self.set_morpher()
        if self.before is None:
            pass
        else:
            self.after = self.my_morpher(self.before)
            self.cv2_pic_to_pixmap(self.before, 'gray', self.label)
            self.cv2_pic_to_pixmap(self.after, 'gray', self.label_2)

    def set_morpher(self):
        kernel1 = cv2.getStructuringElement(
            self.method(self.comboBox_kernel1), (self.spinBox_x1.value(), self.spinBox_y1.value()))
        kernel2 = cv2.getStructuringElement(
            self.method(self.comboBox_kernel2), (self.spinBox_x2.value(), self.spinBox_y2.value()))
        self.morph1 = partial(cv2.morphologyEx,
                        op=self.method(self.comboBox_morph1), kernel=kernel1, iterations=self.spinBox_circ1.value())
        self.morph2 = partial(cv2.morphologyEx,
                        op=self.method(self.comboBox_morph2), kernel=kernel2, iterations=self.spinBox_circ2.value())

    def my_morpher(self, pic):
        return self.morph2(self.morph1(pic))

    def method(self, combobox):
        return self.methods[combobox.currentText()]

    def cv2_pic_to_pixmap(self, img, ptype, label=None):
        # 本函数可将cv2格式的图片转换为QPixmap能作用的格式，并设置图片尺寸为label控件的大小
        if ptype == 'rgb':
            tul = 3 * img.shape[1]
            pic_q_image = QImage(img.data, img.shape[1], img.shape[0], tul, QImage.Format_RGB888)
        elif ptype == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tul = 3 * img.shape[1]
            pic_q_image = QImage(img.data, img.shape[1], img.shape[0], tul, QImage.Format_RGB888)
        elif ptype == 'gray':
            tul = 1 * img.shape[1]
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
    dialog = MorphAdjuster()
    pic = cv2.imread('./.img/car_pic1.jpg')
    # pic = cv2.imread(r'C:\Users\qizidog\Desktop\2.png')
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.threshold(pic, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    dialog.before = pic
    f = dialog.exec_()
    print(f)
    # cv2.imwrite(r'C:\Users\qizidog\Desktop\mor.png', dialog.after)
    # print(dialog.after)
