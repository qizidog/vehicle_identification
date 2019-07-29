# -- coding:utf-8 --
# author: ZQF time:2019/3/7

import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QLabel
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QImage
import ColorThresholderUI


class ColorThresholder(QtWidgets.QDialog, ColorThresholderUI.Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.show()  # 要不要这句不影响？加上这句过后dialog变成了非模态的
        # self.exec_()  # 如果只用这句来执行程序，只有当程序退出后才会执行后面的语句
        self.color, self.mask = None, None
        self.pushButton.clicked.connect(self.set_default)
        self.horizontalSlider_HL.valueChanged.connect(self.show_mask)
        self.horizontalSlider_HH.valueChanged.connect(self.show_mask)
        self.horizontalSlider_SL.valueChanged.connect(self.show_mask)
        self.horizontalSlider_SH.valueChanged.connect(self.show_mask)
        self.horizontalSlider_VL.valueChanged.connect(self.show_mask)
        self.horizontalSlider_VH.valueChanged.connect(self.show_mask)
        self.checkBox.stateChanged.connect(self.show_mask)
        self.show_mask()

    def show_mask(self):
        if self.color is None:
            pass
        else:
            self.mask = self.get_mask(self.color)
            self.cv2_pic_to_pixmap(self.color, 'bgr', self.label)
            self.cv2_pic_to_pixmap(self.mask, 'gray', self.label_2)

    def get_mask(self, bgr):
        h_min = self.horizontalSlider_HL.value()
        h_max = self.horizontalSlider_HH.value()
        s_min = self.horizontalSlider_SL.value()
        s_max = self.horizontalSlider_SH.value()
        v_min = self.horizontalSlider_VL.value()
        v_max = self.horizontalSlider_VH.value()
        reverse = self.checkBox.isChecked()
        I = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        if h_min >= h_max:
            mask_h = (I[:, :, 0] >= h_min) | (I[:, :, 0] <= h_max)
        else:
            mask_h = (I[:, :, 0] >= h_min) & (I[:, :, 0] <= h_max)
        if s_min >= s_max:
            mask_s = (I[:, :, 1] >= s_min) | (I[:, :, 1] <= s_max)
        else:
            mask_s = (I[:, :, 1] >= s_min) & (I[:, :, 1] <= s_max)
        if v_min >= v_max:
            mask_v = (I[:, :, 2] >= v_min) | (I[:, :, 2] <= v_max)
        else:
            mask_v = (I[:, :, 2] >= v_min) & (I[:, :, 2] <= v_max)
        mask = (mask_h & mask_s & mask_v) * 255
        if reverse:
            mask = ~mask
        mask = np.uint8(mask)
        # masked_pic = cv2.bitwise_and(bgr, bgr, mask=mask)
        return mask  # , masked_pic

    def set_default(self):
        self.horizontalSlider_HL.setValue(70)
        self.horizontalSlider_HH.setValue(4)
        self.horizontalSlider_SL.setValue(0)
        self.horizontalSlider_SH.setValue(30)
        self.horizontalSlider_VL.setValue(102)
        self.horizontalSlider_VH.setValue(199)

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
    dialog = ColorThresholder()
    f = dialog.exec_()
    print(f)
    # app.exec_()
    # sys.exit(app.exec_())
