

from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

class Mywindow(QMainWindow):
    def __init__(self):
        super(Mywindow, self).__init__()
        self.setGeometry(200, 200, 300, 300)
        self.setWindowTitle('Application Overview')
        self.initUI()

    def initUI(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText('My First Label!')
        self.label.move(50, 50)
        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText('Click Me!')
        self.b1.clicked.connect(self.clicked)
    def clicked(self):
        self.label.setText('You pressed the button.')
        self.update()
    def update(self):
        self.label.adjustSize()


def window():
    app = QApplication(sys.argv)  #application setup
    win = Mywindow()
    win.show()
    sys.exit(app.exec_())
window()


