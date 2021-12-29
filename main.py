import sys
import tkinter
from typing import Union

from future.moves.tkinter import filedialog
from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QStackedLayout, QLabel, \
    QGridLayout
from PySide2.QtGui import QPixmap
from PySide2.QtCore import Qt
from PIL import Image


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.window_setup()

    def window_setup(self):
        # main window
        self.setWindowTitle("Classify an image")
        self.setFixedSize(700, 650)

        # central widget
        self.central_widget = QWidget()
        self.main_layout = QStackedLayout(self.central_widget)

        # 1st window
        self.page_1 = QWidget()
        self.layout1 = QVBoxLayout(self.page_1)
        self.start_button = QPushButton("Choose image")
        self.start_button.setFixedSize(600, 600)
        self.start_button.pos()
        self.start_button.clicked.connect(self.load_image)
        self.layout1.addWidget(self.start_button, alignment=Qt.AlignCenter)

        # 2nd window
        self.page_2 = QWidget()
        self.layout2 = QGridLayout(self.page_2)
        self.displayed_image = QPixmap()
        self.image_label = QLabel(scaledContents=True)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.classify_button = QPushButton("classify")
        self.another_button = QPushButton("another image")
        self.layout2.setRowMinimumHeight(0, 50)
        self.layout2.setRowMinimumHeight(1, 500)
        self.layout2.addWidget(self.classify_button, 0, 0, alignment=Qt.AlignCenter)
        self.layout2.addWidget(self.another_button, 0, 1, alignment=Qt.AlignCenter)
        self.layout2.addWidget(self.image_label, 1, 0, 1, 2, alignment=Qt.AlignCenter)

        # setting up main layout
        self.main_layout.addWidget(self.page_1)
        self.main_layout.addWidget(self.page_2)
        self.setCentralWidget(self.central_widget)
        self.page_1.show()
        self.page_2.hide()

    def load_image(self):
        # get path of image with filedialog
        root_window = tkinter.Tk()
        root_window.withdraw()
        root_window.update()
        image_path = filedialog.askopenfilename(initialdir='/home/bartek/Pictures')
        root_window.destroy()
        if image_path:
            self.displayed_image.load(image_path)
            self.image_label.setPixmap(self.displayed_image)
            self.image_label.setAlignment(Qt.AlignCenter)
            self.page_2.show()
            self.page_1.hide()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
