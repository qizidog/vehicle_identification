# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MorphAdjusterUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1076, 449)
        self.gridLayout_3 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setMinimumSize(QtCore.QSize(751, 171))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setMinimumSize(QtCore.QSize(751, 171))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.gridLayout_3.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_11 = QtWidgets.QLabel(Dialog)
        self.label_11.setMaximumSize(QtCore.QSize(81, 18))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_2.addWidget(self.label_11)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setMinimumSize(QtCore.QSize(71, 21))
        self.label_3.setMaximumSize(QtCore.QSize(71, 21))
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.comboBox_kernel1 = QtWidgets.QComboBox(Dialog)
        self.comboBox_kernel1.setMinimumSize(QtCore.QSize(101, 24))
        self.comboBox_kernel1.setMaximumSize(QtCore.QSize(200, 24))
        self.comboBox_kernel1.setObjectName("comboBox_kernel1")
        self.comboBox_kernel1.addItem("")
        self.comboBox_kernel1.addItem("")
        self.comboBox_kernel1.addItem("")
        self.gridLayout.addWidget(self.comboBox_kernel1, 0, 1, 1, 2)
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setMinimumSize(QtCore.QSize(75, 18))
        self.label_4.setMaximumSize(QtCore.QSize(75, 18))
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)
        self.spinBox_x1 = QtWidgets.QSpinBox(Dialog)
        self.spinBox_x1.setMinimumSize(QtCore.QSize(49, 25))
        self.spinBox_x1.setMaximumSize(QtCore.QSize(55, 25))
        self.spinBox_x1.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_x1.setMinimum(1)
        self.spinBox_x1.setMaximum(15)
        self.spinBox_x1.setSingleStep(2)
        self.spinBox_x1.setObjectName("spinBox_x1")
        self.gridLayout.addWidget(self.spinBox_x1, 1, 1, 1, 1)
        self.spinBox_y1 = QtWidgets.QSpinBox(Dialog)
        self.spinBox_y1.setMinimumSize(QtCore.QSize(49, 25))
        self.spinBox_y1.setMaximumSize(QtCore.QSize(55, 25))
        self.spinBox_y1.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_y1.setMinimum(1)
        self.spinBox_y1.setMaximum(15)
        self.spinBox_y1.setSingleStep(2)
        self.spinBox_y1.setObjectName("spinBox_y1")
        self.gridLayout.addWidget(self.spinBox_y1, 1, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setMinimumSize(QtCore.QSize(81, 18))
        self.label_5.setMaximumSize(QtCore.QSize(81, 18))
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 1)
        self.comboBox_morph1 = QtWidgets.QComboBox(Dialog)
        self.comboBox_morph1.setObjectName("comboBox_morph1")
        self.comboBox_morph1.addItem("")
        self.comboBox_morph1.addItem("")
        self.comboBox_morph1.addItem("")
        self.comboBox_morph1.addItem("")
        self.gridLayout.addWidget(self.comboBox_morph1, 2, 1, 1, 2)
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setMinimumSize(QtCore.QSize(81, 18))
        self.label_6.setMaximumSize(QtCore.QSize(81, 18))
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 3, 0, 1, 1)
        self.spinBox_circ1 = QtWidgets.QSpinBox(Dialog)
        self.spinBox_circ1.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.spinBox_circ1.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_circ1.setMinimum(1)
        self.spinBox_circ1.setMaximum(5)
        self.spinBox_circ1.setObjectName("spinBox_circ1")
        self.gridLayout.addWidget(self.spinBox_circ1, 3, 1, 1, 2)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.line = QtWidgets.QFrame(Dialog)
        self.line.setMinimumSize(QtCore.QSize(286, 3))
        self.line.setMaximumSize(QtCore.QSize(286, 3))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.label_12 = QtWidgets.QLabel(Dialog)
        self.label_12.setMaximumSize(QtCore.QSize(81, 18))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_2.addWidget(self.label_12)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setMinimumSize(QtCore.QSize(71, 21))
        self.label_7.setMaximumSize(QtCore.QSize(71, 21))
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 0, 0, 1, 1)
        self.comboBox_kernel2 = QtWidgets.QComboBox(Dialog)
        self.comboBox_kernel2.setMinimumSize(QtCore.QSize(101, 24))
        self.comboBox_kernel2.setMaximumSize(QtCore.QSize(200, 24))
        self.comboBox_kernel2.setObjectName("comboBox_kernel2")
        self.comboBox_kernel2.addItem("")
        self.comboBox_kernel2.addItem("")
        self.comboBox_kernel2.addItem("")
        self.gridLayout_2.addWidget(self.comboBox_kernel2, 0, 1, 1, 2)
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setMinimumSize(QtCore.QSize(75, 18))
        self.label_8.setMaximumSize(QtCore.QSize(75, 18))
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 1, 0, 1, 1)
        self.spinBox_x2 = QtWidgets.QSpinBox(Dialog)
        self.spinBox_x2.setMinimumSize(QtCore.QSize(49, 25))
        self.spinBox_x2.setMaximumSize(QtCore.QSize(55, 25))
        self.spinBox_x2.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_x2.setMinimum(1)
        self.spinBox_x2.setMaximum(15)
        self.spinBox_x2.setSingleStep(2)
        self.spinBox_x2.setObjectName("spinBox_x2")
        self.gridLayout_2.addWidget(self.spinBox_x2, 1, 1, 1, 1)
        self.spinBox_y2 = QtWidgets.QSpinBox(Dialog)
        self.spinBox_y2.setMinimumSize(QtCore.QSize(49, 25))
        self.spinBox_y2.setMaximumSize(QtCore.QSize(55, 25))
        self.spinBox_y2.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_y2.setMinimum(1)
        self.spinBox_y2.setMaximum(15)
        self.spinBox_y2.setSingleStep(2)
        self.spinBox_y2.setObjectName("spinBox_y2")
        self.gridLayout_2.addWidget(self.spinBox_y2, 1, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setMinimumSize(QtCore.QSize(81, 18))
        self.label_9.setMaximumSize(QtCore.QSize(81, 18))
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 2, 0, 1, 1)
        self.comboBox_morph2 = QtWidgets.QComboBox(Dialog)
        self.comboBox_morph2.setObjectName("comboBox_morph2")
        self.comboBox_morph2.addItem("")
        self.comboBox_morph2.addItem("")
        self.comboBox_morph2.addItem("")
        self.comboBox_morph2.addItem("")
        self.gridLayout_2.addWidget(self.comboBox_morph2, 2, 1, 1, 2)
        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_10.setMinimumSize(QtCore.QSize(81, 18))
        self.label_10.setMaximumSize(QtCore.QSize(81, 18))
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 3, 0, 1, 1)
        self.spinBox_circ2 = QtWidgets.QSpinBox(Dialog)
        self.spinBox_circ2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.spinBox_circ2.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_circ2.setMinimum(1)
        self.spinBox_circ2.setMaximum(5)
        self.spinBox_circ2.setObjectName("spinBox_circ2")
        self.gridLayout_2.addWidget(self.spinBox_circ2, 3, 1, 1, 2)
        self.verticalLayout_2.addLayout(self.gridLayout_2)
        self.line_2 = QtWidgets.QFrame(Dialog)
        self.line_2.setMinimumSize(QtCore.QSize(286, 3))
        self.line_2.setMaximumSize(QtCore.QSize(286, 3))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_2.addWidget(self.line_2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setMaximumSize(QtCore.QSize(286, 34))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout_2.addWidget(self.buttonBox)
        self.gridLayout_3.addLayout(self.verticalLayout_2, 0, 1, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_11.setText(_translate("Dialog", " 运算1"))
        self.label_3.setText(_translate("Dialog", "运算核："))
        self.comboBox_kernel1.setItemText(0, _translate("Dialog", "矩形"))
        self.comboBox_kernel1.setItemText(1, _translate("Dialog", "椭圆形"))
        self.comboBox_kernel1.setItemText(2, _translate("Dialog", "十字形"))
        self.label_4.setText(_translate("Dialog", "尺寸X,Y："))
        self.label_5.setText(_translate("Dialog", "运算："))
        self.comboBox_morph1.setItemText(0, _translate("Dialog", "开运算"))
        self.comboBox_morph1.setItemText(1, _translate("Dialog", "闭运算"))
        self.comboBox_morph1.setItemText(2, _translate("Dialog", "膨胀"))
        self.comboBox_morph1.setItemText(3, _translate("Dialog", "腐蚀"))
        self.label_6.setText(_translate("Dialog", "迭代次数："))
        self.label_12.setText(_translate("Dialog", " 运算2"))
        self.label_7.setText(_translate("Dialog", "运算核："))
        self.comboBox_kernel2.setItemText(0, _translate("Dialog", "矩形"))
        self.comboBox_kernel2.setItemText(1, _translate("Dialog", "椭圆形"))
        self.comboBox_kernel2.setItemText(2, _translate("Dialog", "十字形"))
        self.label_8.setText(_translate("Dialog", "尺寸X,Y："))
        self.label_9.setText(_translate("Dialog", "运算："))
        self.comboBox_morph2.setItemText(0, _translate("Dialog", "开运算"))
        self.comboBox_morph2.setItemText(1, _translate("Dialog", "闭运算"))
        self.comboBox_morph2.setItemText(2, _translate("Dialog", "膨胀"))
        self.comboBox_morph2.setItemText(3, _translate("Dialog", "腐蚀"))
        self.label_10.setText(_translate("Dialog", "迭代次数："))
