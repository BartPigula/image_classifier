import sys
import tkinter
from future.moves.tkinter import filedialog
from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QStackedLayout, QLabel, \
    QGridLayout
from PySide2.QtGui import QPixmap
from PySide2.QtCore import Qt
from PIL import Image
import numpy as np
from keras.models import load_model


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.window_setup()

    def window_setup(self):
        # main window
        self.setWindowTitle("Classify an image")
        self.setFixedSize(650, 550)

        # central widget
        self.central_widget = QWidget()
        self.main_layout = QStackedLayout(self.central_widget)

        # 1st window
        self.page_1 = QWidget()
        self.layout1 = QVBoxLayout(self.page_1)
        self.start_button = QPushButton("Choose image")
        self.start_button.setFixedSize(500, 500)
        self.start_button.pos()
        self.start_button.clicked.connect(self.load_image)
        self.layout1.addWidget(self.start_button, alignment=Qt.AlignCenter)

        # 2nd window
        self.page_2 = QWidget()
        self.layout2 = QVBoxLayout(self.page_2)
        self.displayed_image = QPixmap()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignHCenter)
        self.classify_button = QPushButton("classify")
        self.another_button = QPushButton("another image")
        self.result = QLabel()

        self.another_button.clicked.connect(self.load_image)
        self.classify_button.clicked.connect(self.classify_image)

        self.layout2.addWidget(self.classify_button, alignment=Qt.AlignCenter)
        self.layout2.addWidget(self.another_button, alignment=Qt.AlignCenter)
        self.layout2.addWidget(self.image_label, alignment=Qt.AlignHCenter)
        self.layout2.addWidget(self.result, alignment=Qt.AlignCenter)

        # setting up main layout
        self.main_layout.addWidget(self.page_1)
        self.main_layout.addWidget(self.page_2)
        self.setCentralWidget(self.central_widget)
        self.page_1.show()
        self.page_2.hide()

        # neural net model
        self.model = load_model('/home/bartek/PycharmProjects/image_classifier/net_model_0.684400.h5')

    def load_image(self):
        # get path of image with filedialog
        root_window = tkinter.Tk()
        root_window.withdraw()
        root_window.update()
        image_path = filedialog.askopenfilename(initialdir='/home/bartek/Pictures')
        root_window.destroy()
        if image_path:
            self.image_to_classify = Image.open(image_path)
            self.displayed_image.load(image_path)
            size = self.displayed_image.size()
            if size.width() > 500 or size.height() > 500:
                self.displayed_image = self.displayed_image.scaled(500, 500, Qt.IgnoreAspectRatio)
            self.image_label.clear()
            self.image_label.setPixmap(self.displayed_image)
            self.page_2.show()
            self.page_1.hide()
        self.result.clear()

    def prepare_image(self):
        self.image_to_classify = self.image_to_classify.resize((32, 32))
        self.image_to_classify = np.expand_dims(self.image_to_classify, axis=0)
        self.image_to_classify = np.array(self.image_to_classify)

    def classify_image(self):
        class_dictionary = {
            0: 'aeroplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }
        self.prepare_image()
        prediction = np.argmax(self.model.predict(self.image_to_classify), axis=1)
        self.result.setText(class_dictionary[prediction[0]])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
