# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ColorThresholderUI.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1072, 569)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Dialog)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setMaximumSize(QtCore.QSize(281, 391))
        self.widget.setObjectName("widget")
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setGeometry(QtCore.QRect(30, 340, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setGeometry(QtCore.QRect(60, 10, 31, 21))
        self.label_6.setFrameShape(QtWidgets.QFrame.Box)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.spinBox_3 = QtWidgets.QSpinBox(self.widget)
        self.spinBox_3.setGeometry(QtCore.QRect(230, 140, 42, 22))
        self.spinBox_3.setMaximum(255)
        self.spinBox_3.setProperty("value", 255)
        self.spinBox_3.setObjectName("spinBox_3")
        self.label_8 = QtWidgets.QLabel(self.widget)
        self.label_8.setGeometry(QtCore.QRect(60, 140, 31, 21))
        self.label_8.setFrameShape(QtWidgets.QFrame.Box)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_7 = QtWidgets.QLabel(self.widget)
        self.label_7.setGeometry(QtCore.QRect(60, 50, 31, 21))
        self.label_7.setFrameShape(QtWidgets.QFrame.Box)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_11 = QtWidgets.QLabel(self.widget)
        self.label_11.setGeometry(QtCore.QRect(60, 240, 31, 21))
        self.label_11.setFrameShape(QtWidgets.QFrame.Box)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(10, 30, 31, 21))
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_9 = QtWidgets.QLabel(self.widget)
        self.label_9.setGeometry(QtCore.QRect(60, 100, 31, 21))
        self.label_9.setFrameShape(QtWidgets.QFrame.Box)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.horizontalSlider_HL = QtWidgets.QSlider(self.widget)
        self.horizontalSlider_HL.setGeometry(QtCore.QRect(110, 12, 111, 16))
        self.horizontalSlider_HL.setMaximum(180)
        self.horizontalSlider_HL.setProperty("value", 0)
        self.horizontalSlider_HL.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_HL.setObjectName("horizontalSlider_HL")
        self.label_10 = QtWidgets.QLabel(self.widget)
        self.label_10.setGeometry(QtCore.QRect(60, 200, 31, 21))
        self.label_10.setFrameShape(QtWidgets.QFrame.Box)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.horizontalSlider_VL = QtWidgets.QSlider(self.widget)
        self.horizontalSlider_VL.setGeometry(QtCore.QRect(110, 202, 111, 16))
        self.horizontalSlider_VL.setMaximum(255)
        self.horizontalSlider_VL.setProperty("value", 0)
        self.horizontalSlider_VL.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_VL.setObjectName("horizontalSlider_VL")
        self.spinBox_5 = QtWidgets.QSpinBox(self.widget)
        self.spinBox_5.setGeometry(QtCore.QRect(230, 240, 42, 22))
        self.spinBox_5.setMaximum(255)
        self.spinBox_5.setProperty("value", 255)
        self.spinBox_5.setObjectName("spinBox_5")
        self.horizontalSlider_VH = QtWidgets.QSlider(self.widget)
        self.horizontalSlider_VH.setGeometry(QtCore.QRect(110, 240, 111, 16))
        self.horizontalSlider_VH.setMaximum(255)
        self.horizontalSlider_VH.setProperty("value", 255)
        self.horizontalSlider_VH.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_VH.setInvertedAppearance(False)
        self.horizontalSlider_VH.setObjectName("horizontalSlider_VH")
        self.spinBox_4 = QtWidgets.QSpinBox(self.widget)
        self.spinBox_4.setGeometry(QtCore.QRect(230, 95, 42, 22))
        self.spinBox_4.setMaximum(255)
        self.spinBox_4.setObjectName("spinBox_4")
        self.spinBox_6 = QtWidgets.QSpinBox(self.widget)
        self.spinBox_6.setGeometry(QtCore.QRect(230, 195, 42, 22))
        self.spinBox_6.setMaximum(255)
        self.spinBox_6.setObjectName("spinBox_6")
        self.horizontalSlider_SH = QtWidgets.QSlider(self.widget)
        self.horizontalSlider_SH.setGeometry(QtCore.QRect(110, 140, 111, 16))
        self.horizontalSlider_SH.setMaximum(255)
        self.horizontalSlider_SH.setProperty("value", 255)
        self.horizontalSlider_SH.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_SH.setInvertedAppearance(False)
        self.horizontalSlider_SH.setObjectName("horizontalSlider_SH")
        self.horizontalSlider_SL = QtWidgets.QSlider(self.widget)
        self.horizontalSlider_SL.setGeometry(QtCore.QRect(110, 102, 111, 16))
        self.horizontalSlider_SL.setMaximum(255)
        self.horizontalSlider_SL.setProperty("value", 0)
        self.horizontalSlider_SL.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_SL.setObjectName("horizontalSlider_SL")
        self.spinBox = QtWidgets.QSpinBox(self.widget)
        self.spinBox.setGeometry(QtCore.QRect(230, 5, 42, 22))
        self.spinBox.setMaximum(180)
        self.spinBox.setObjectName("spinBox")
        self.spinBox_2 = QtWidgets.QSpinBox(self.widget)
        self.spinBox_2.setGeometry(QtCore.QRect(230, 50, 42, 22))
        self.spinBox_2.setMaximum(180)
        self.spinBox_2.setProperty("value", 180)
        self.spinBox_2.setObjectName("spinBox_2")
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setGeometry(QtCore.QRect(10, 220, 31, 21))
        self.label_5.setFrameShape(QtWidgets.QFrame.Box)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.horizontalSlider_HH = QtWidgets.QSlider(self.widget)
        self.horizontalSlider_HH.setGeometry(QtCore.QRect(110, 50, 111, 16))
        self.horizontalSlider_HH.setMaximum(180)
        self.horizontalSlider_HH.setProperty("value", 180)
        self.horizontalSlider_HH.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_HH.setInvertedAppearance(False)
        self.horizontalSlider_HH.setObjectName("horizontalSlider_HH")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setGeometry(QtCore.QRect(10, 120, 31, 21))
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.buttonBox = QtWidgets.QDialogButtonBox(self.widget)
        self.buttonBox.setGeometry(QtCore.QRect(120, 340, 156, 23))
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.checkBox = QtWidgets.QCheckBox(self.widget)
        self.checkBox.setGeometry(QtCore.QRect(30, 290, 105, 22))
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout.addWidget(self.widget)

        self.retranslateUi(Dialog)
        self.horizontalSlider_HL.valueChanged['int'].connect(self.spinBox.setValue)
        self.spinBox.valueChanged['int'].connect(self.horizontalSlider_HL.setValue)
        self.horizontalSlider_HH.valueChanged['int'].connect(self.spinBox_2.setValue)
        self.spinBox_2.valueChanged['int'].connect(self.horizontalSlider_HH.setValue)
        self.horizontalSlider_SL.valueChanged['int'].connect(self.spinBox_4.setValue)
        self.spinBox_4.valueChanged['int'].connect(self.horizontalSlider_SL.setValue)
        self.horizontalSlider_SH.valueChanged['int'].connect(self.spinBox_3.setValue)
        self.spinBox_3.valueChanged['int'].connect(self.horizontalSlider_SH.setValue)
        self.horizontalSlider_VL.valueChanged['int'].connect(self.spinBox_6.setValue)
        self.spinBox_6.valueChanged['int'].connect(self.horizontalSlider_VL.setValue)
        self.horizontalSlider_VH.valueChanged['int'].connect(self.spinBox_5.setValue)
        self.spinBox_5.valueChanged['int'].connect(self.horizontalSlider_VH.setValue)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Default"))
        self.label_6.setText(_translate("Dialog", "low"))
        self.label_8.setText(_translate("Dialog", "high"))
        self.label_7.setText(_translate("Dialog", "high"))
        self.label_11.setText(_translate("Dialog", "high"))
        self.label_3.setText(_translate("Dialog", "H"))
        self.label_9.setText(_translate("Dialog", "low"))
        self.label_10.setText(_translate("Dialog", "low"))
        self.label_5.setText(_translate("Dialog", "V"))
        self.label_4.setText(_translate("Dialog", "S"))
        self.checkBox.setText(_translate("Dialog", "Reverse"))


