# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'motionVectorUI.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1538, 813)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_frameBox = QtWidgets.QLabel(self.centralwidget)
        self.label_frameBox.setMinimumSize(QtCore.QSize(1021, 205))
        self.label_frameBox.setSizeIncrement(QtCore.QSize(100, 60))
        self.label_frameBox.setBaseSize(QtCore.QSize(1021, 205))
        self.label_frameBox.setAutoFillBackground(False)
        self.label_frameBox.setFrameShape(QtWidgets.QFrame.Box)
        self.label_frameBox.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_frameBox.setText("")
        self.label_frameBox.setScaledContents(False)
        self.label_frameBox.setObjectName("label_frameBox")
        self.verticalLayout.addWidget(self.label_frameBox)
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setMinimumSize(QtCore.QSize(1021, 205))
        self.label2.setSizeIncrement(QtCore.QSize(100, 60))
        self.label2.setBaseSize(QtCore.QSize(1021, 205))
        self.label2.setAutoFillBackground(False)
        self.label2.setFrameShape(QtWidgets.QFrame.Box)
        self.label2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label2.setText("")
        self.label2.setScaledContents(False)
        self.label2.setObjectName("label2")
        self.verticalLayout.addWidget(self.label2)
        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setMinimumSize(QtCore.QSize(1021, 205))
        self.label3.setSizeIncrement(QtCore.QSize(500, 60))
        self.label3.setBaseSize(QtCore.QSize(1021, 205))
        self.label3.setAutoFillBackground(False)
        self.label3.setFrameShape(QtWidgets.QFrame.Box)
        self.label3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label3.setText("")
        self.label3.setScaledContents(False)
        self.label3.setObjectName("label3")
        self.verticalLayout.addWidget(self.label3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_play_pause = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_play_pause.setMinimumSize(QtCore.QSize(25, 25))
        self.pushButton_play_pause.setMaximumSize(QtCore.QSize(25, 25))
        self.pushButton_play_pause.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_play_pause.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/menu/pause.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon.addPixmap(QtGui.QPixmap(":/menu/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        icon.addPixmap(QtGui.QPixmap(":/menu/pause.png"), QtGui.QIcon.Selected, QtGui.QIcon.Off)
        icon.addPixmap(QtGui.QPixmap(":/menu/play.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.pushButton_play_pause.setIcon(icon)
        self.pushButton_play_pause.setCheckable(True)
        self.pushButton_play_pause.setChecked(True)
        self.pushButton_play_pause.setDefault(False)
        self.pushButton_play_pause.setObjectName("pushButton_play_pause")
        self.horizontalLayout.addWidget(self.pushButton_play_pause)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setMaximumSize(QtCore.QSize(16777215, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout.addWidget(self.horizontalSlider)
        self.label_cur_num = QtWidgets.QLabel(self.centralwidget)
        self.label_cur_num.setMaximumSize(QtCore.QSize(16777215, 25))
        self.label_cur_num.setObjectName("label_cur_num")
        self.horizontalLayout.addWidget(self.label_cur_num)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setMaximum(2000)
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout.addWidget(self.spinBox)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.spinBox_2 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_2.setMaximum(2000)
        self.spinBox_2.setObjectName("spinBox_2")
        self.horizontalLayout.addWidget(self.spinBox_2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setMaximumSize(QtCore.QSize(60, 16777215))
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setMinimumSize(QtCore.QSize(431, 0))
        self.tableWidget.setMaximumSize(QtCore.QSize(600, 16777215))
        self.tableWidget.setAutoFillBackground(False)
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableWidget.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(6)
        self.tableWidget.setRowCount(5)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(5, item)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setHighlightSections(True)
        self.tableWidget.horizontalHeader().setSortIndicatorShown(False)
        self.horizontalLayout_2.addWidget(self.tableWidget)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1538, 30))
        self.menubar.setObjectName("menubar")
        self.menu_file = QtWidgets.QMenu(self.menubar)
        self.menu_file.setObjectName("menu_file")
        self.menu_edit = QtWidgets.QMenu(self.menubar)
        self.menu_edit.setObjectName("menu_edit")
        self.menu_help = QtWidgets.QMenu(self.menubar)
        self.menu_help.setObjectName("menu_help")
        self.menu_filter = QtWidgets.QMenu(self.menubar)
        self.menu_filter.setObjectName("menu_filter")
        MainWindow.setMenuBar(self.menubar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setMouseTracking(True)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.toolBar_2 = QtWidgets.QToolBar(MainWindow)
        self.toolBar_2.setMouseTracking(True)
        self.toolBar_2.setObjectName("toolBar_2")
        MainWindow.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBar_2)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.action_open = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/menu/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_open.setIcon(icon1)
        self.action_open.setObjectName("action_open")
        self.action_exit = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/menu/exit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_exit.setIcon(icon2)
        self.action_exit.setObjectName("action_exit")
        self.action_undo = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/menu/undo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_undo.setIcon(icon3)
        self.action_undo.setObjectName("action_undo")
        self.action_redo = QtWidgets.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/menu/redo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_redo.setIcon(icon4)
        self.action_redo.setObjectName("action_redo")
        self.action_rotate = QtWidgets.QAction(MainWindow)
        self.action_rotate.setCheckable(True)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/menu/rotate.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_rotate.setIcon(icon5)
        self.action_rotate.setObjectName("action_rotate")
        self.action_viewCap = QtWidgets.QAction(MainWindow)
        self.action_viewCap.setCheckable(True)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/menu/显示区.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_viewCap.setIcon(icon6)
        self.action_viewCap.setObjectName("action_viewCap")
        self.action_dataCap = QtWidgets.QAction(MainWindow)
        self.action_dataCap.setCheckable(True)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/menu/统计区.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_dataCap.setIcon(icon7)
        self.action_dataCap.setObjectName("action_dataCap")
        self.action_save = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/menu/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_save.setIcon(icon8)
        self.action_save.setObjectName("action_save")
        self.action_ColorThreshholder = QtWidgets.QAction(MainWindow)
        self.action_ColorThreshholder.setObjectName("action_ColorThreshholder")
        self.action_ContourFilter = QtWidgets.QAction(MainWindow)
        self.action_ContourFilter.setObjectName("action_ContourFilter")
        self.action_MorphologyEx = QtWidgets.QAction(MainWindow)
        self.action_MorphologyEx.setObjectName("action_MorphologyEx")
        self.action_Blur = QtWidgets.QAction(MainWindow)
        self.action_Blur.setObjectName("action_Blur")
        self.menu_file.addAction(self.action_open)
        self.menu_file.addAction(self.action_save)
        self.menu_file.addAction(self.action_exit)
        self.menu_edit.addAction(self.action_undo)
        self.menu_edit.addAction(self.action_redo)
        self.menu_edit.addSeparator()
        self.menu_edit.addAction(self.action_rotate)
        self.menu_edit.addAction(self.action_viewCap)
        self.menu_edit.addAction(self.action_dataCap)
        self.menu_filter.addAction(self.action_ColorThreshholder)
        self.menu_filter.addAction(self.action_ContourFilter)
        self.menu_filter.addAction(self.action_MorphologyEx)
        self.menu_filter.addAction(self.action_Blur)
        self.menubar.addAction(self.menu_file.menuAction())
        self.menubar.addAction(self.menu_edit.menuAction())
        self.menubar.addAction(self.menu_filter.menuAction())
        self.menubar.addAction(self.menu_help.menuAction())
        self.toolBar.addAction(self.action_open)
        self.toolBar.addAction(self.action_save)
        self.toolBar.addAction(self.action_exit)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.action_undo)
        self.toolBar.addAction(self.action_redo)
        self.toolBar_2.addAction(self.action_rotate)
        self.toolBar_2.addAction(self.action_viewCap)
        self.toolBar_2.addAction(self.action_dataCap)
        self.toolBar_2.addSeparator()

        self.retranslateUi(MainWindow)
        self.action_exit.triggered.connect(MainWindow.close)
        self.horizontalSlider.valueChanged['int'].connect(self.label_cur_num.setNum)
        self.horizontalSlider.valueChanged['int'].connect(self.spinBox.setValue)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_play_pause.setShortcut(_translate("MainWindow", "Space"))
        self.label.setText(_translate("MainWindow", "Total frame："))
        self.label_2.setText(_translate("MainWindow", "0"))
        self.label_cur_num.setText(_translate("MainWindow", "∞"))
        self.label_3.setText(_translate("MainWindow", "frame 1："))
        self.label_4.setText(_translate("MainWindow", "frame 2："))
        self.pushButton.setText(_translate("MainWindow", "ok"))
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "1"))
        item = self.tableWidget.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "2"))
        item = self.tableWidget.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "3"))
        item = self.tableWidget.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "4"))
        item = self.tableWidget.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "5"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "first_pos"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "second_pos"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "movement"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "L"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "area1"))
        item = self.tableWidget.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "area2"))
        self.menu_file.setTitle(_translate("MainWindow", "文件"))
        self.menu_edit.setTitle(_translate("MainWindow", "编辑"))
        self.menu_help.setTitle(_translate("MainWindow", "帮助"))
        self.menu_filter.setTitle(_translate("MainWindow", "筛选"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.toolBar_2.setWindowTitle(_translate("MainWindow", "toolBar_2"))
        self.action_open.setText(_translate("MainWindow", "打开视频"))
        self.action_exit.setText(_translate("MainWindow", "退出"))
        self.action_undo.setText(_translate("MainWindow", "撤回"))
        self.action_undo.setShortcut(_translate("MainWindow", "Ctrl+Z"))
        self.action_redo.setText(_translate("MainWindow", "恢复"))
        self.action_redo.setShortcut(_translate("MainWindow", "Ctrl+Y"))
        self.action_rotate.setText(_translate("MainWindow", "旋转画面"))
        self.action_viewCap.setText(_translate("MainWindow", "截取显示区"))
        self.action_dataCap.setText(_translate("MainWindow", "选择统计区"))
        self.action_save.setText(_translate("MainWindow", "保存"))
        self.action_save.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.action_ColorThreshholder.setText(_translate("MainWindow", "色彩分割"))
        self.action_ContourFilter.setText(_translate("MainWindow", "轮廓筛选"))
        self.action_MorphologyEx.setText(_translate("MainWindow", "形态学处理"))
        self.action_Blur.setText(_translate("MainWindow", "模糊处理"))


import car_icon_rc
