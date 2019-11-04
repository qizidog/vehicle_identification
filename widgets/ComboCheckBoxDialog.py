# -- coding: utf-8 --
# author: ZQF  time: 2019/5/12 13:48

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys


class QComboCheckBoxDialog(QtWidgets.QDialog):
    def __init__(self, title='ComboCheckBoxDialog', label='label', items=[], max=10, parent=None):
        super(QComboCheckBoxDialog, self).__init__(parent)
        self.input_title = title
        self.input_label = label
        self.items = items
        self.max = max
        self.setupUi(self)

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setFixedSize(400, 272)
        Dialog.setSizeGripEnabled(False)
        Dialog.setModal(True)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(20, 40, 281, 21))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        # self.comboBox = QtWidgets.QComboBox(self.frame)
        self.comboBox = ComboCheckBox(self.items, self.max, self.frame)
        self.comboBox.setGeometry(QtCore.QRect(40, 100, 301, 31))
        self.comboBox.setObjectName("comboBox")
        self.buttonBox = QtWidgets.QDialogButtonBox(self.frame)
        self.buttonBox.setGeometry(QtCore.QRect(170, 181, 171, 31))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", self.input_title))
        self.label.setText(_translate("Dialog", self.input_label))


class ComboCheckBox(QtWidgets.QComboBox):
    def __init__(self, items, max=10, parent=None):  # items==[str,str...]
        super(ComboCheckBox, self).__init__(parent)
        self.items = items
        self.qCheckBox = []
        self.qLineEdit = QtWidgets.QLineEdit()
        self.qLineEdit.setReadOnly(True)
        qListWidget = QtWidgets.QListWidget()
        self.ret = ''  # 用来返回的选中的值
        self.max = max

        self.row_num = len(self.items)
        for i in range(self.row_num):
            self.qCheckBox.append(QtWidgets.QCheckBox())
            qItem = QtWidgets.QListWidgetItem(qListWidget)
            self.qCheckBox[i].setText(self.items[i])
            qListWidget.setItemWidget(qItem, self.qCheckBox[i])
            self.qCheckBox[i].stateChanged.connect(self.show)

        self.setLineEdit(self.qLineEdit)
        self.setModel(qListWidget.model())
        self.setView(qListWidget)

    def Selectlist(self):
        Outputlist = []
        for i in range(self.row_num):
            if self.qCheckBox[i].isChecked() == True:
                Outputlist.append(self.qCheckBox[i].text())
        return Outputlist

    def show(self):
        show = ''
        self.qLineEdit.setReadOnly(False)
        self.qLineEdit.clear()
        for i, item in enumerate(self.Selectlist()):
            if i == 0:
                show += item
            elif i <= self.max:
                show += ','+item
            else:
                QtWidgets.QMessageBox.information(self, '提示', '最多选择%s个选项' % self.max, QtWidgets.QMessageBox.Ok)
                self.qCheckBox[i].setChecked(False)
        self.qLineEdit.setText(show)
        self.qLineEdit.setReadOnly(True)
        self.ret = self.qLineEdit.text()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = QComboCheckBoxDialog('轨迹选择', '请选择需要绘制轨迹的车辆标号：', ['1', '2', '3','4', '5', '6'], 5)
    a = dialog.exec_()
    if a:
        items = dialog.comboBox.ret
        print(items)
    # sys.exit(app.exec_())
