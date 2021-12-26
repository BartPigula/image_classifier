import sys

from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.window_setup()

    def window_setup(self):
        self.setWindowTitle("Classify an image")
        self.setMinimumSize(500, 500)
        self.setMaximumSize(1000, 700)
        start_button = QPushButton("Choose image")
        self.setCentralWidget(start_button)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
