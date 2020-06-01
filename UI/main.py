# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setGeometry(QtCore.QRect(270, 20, 261, 101))
        self.logo.setText("")
        self.logo.setPixmap(QtGui.QPixmap("../Resources/Super logo.png"))
        self.logo.setScaledContents(True)
        self.logo.setObjectName("logo")
        self.autonomous_button = QtWidgets.QPushButton(self.centralwidget)
        self.autonomous_button.setGeometry(QtCore.QRect(310, 200, 191, 71))
        self.autonomous_button.setStyleSheet("background-color: rgb(83, 83, 83);\n"
"color: rgb(255, 255, 255);\n"
"font: 12pt \"Impact\";")
        self.autonomous_button.setObjectName("autonomous_button")
        self.calib_button = QtWidgets.QPushButton(self.centralwidget)
        self.calib_button.setGeometry(QtCore.QRect(310, 330, 191, 71))
        self.calib_button.setStyleSheet("background-color: rgb(83, 83, 83);\n"
"color: rgb(255, 255, 255);\n"
"font: 12pt \"Impact\";")
        self.calib_button.setObjectName("calib_button")
        self.exit_button = QtWidgets.QPushButton(self.centralwidget)
        self.exit_button.setGeometry(QtCore.QRect(310, 460, 191, 71))
        self.exit_button.setStyleSheet("background-color: rgb(83, 83, 83);\n"
"color: rgb(255, 255, 255);\n"
"font: 12pt \"Impact\";")
        self.exit_button.setObjectName("exit_button")
        self.about_button = QtWidgets.QPushButton(self.centralwidget)
        self.about_button.setGeometry(QtCore.QRect(670, 500, 111, 31))
        self.about_button.setStyleSheet("background-color: rgb(83, 83, 83);\n"
"color: rgb(255, 255, 255);\n"
"font: 8pt \"Impact\";")
        self.about_button.setObjectName("about_button")
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(310, 120, 211, 71))
        self.radioButton.setStyleSheet("font: 10pt \"Impact\";")
        self.radioButton.setObjectName("radioButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.autonomous_button.setStatusTip(_translate("MainWindow", "Entrar en vuelo autónomo que seguira el objeto a detectar"))
        self.autonomous_button.setText(_translate("MainWindow", "Modo Autónomo"))
        self.calib_button.setStatusTip(_translate("MainWindow", "Entrar en modo de calibración para ajustar valores a detectar objeto deseado"))
        self.calib_button.setText(_translate("MainWindow", "Calibración"))
        self.exit_button.setStatusTip(_translate("MainWindow", "Salir del programa"))
        self.exit_button.setText(_translate("MainWindow", "Salir"))
        self.about_button.setStatusTip(_translate("MainWindow", "Documentación del programa"))
        self.about_button.setText(_translate("MainWindow", "Mas Info"))
        self.radioButton.setText(_translate("MainWindow", "Guardar video de sesión"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
