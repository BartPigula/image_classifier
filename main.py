import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import os

# importing needed model parameters
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizer_v1 import SGD
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils.np_utils import to_categorical

# loading training and testing data
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# changing type to float and normalizing data
train_x = train_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255

# performing one-hot encoding for classes - mapping class to integer (1 on the index of that integer)
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)
# getting total amount of classes
classes_count = train_y.shape[1]

# creating model
input_shape = (32, 32, 3)
net = Sequential()
net.add(Conv2D(32, 3, input_shape=input_shape, activation='relu', padding='same', kernel_constraint=maxnorm(3)))
net.add(Dropout(0.1))  # arg between 0 and 1
net.add(Conv2D(32, 3, activation='relu', padding='same', kernel_constraint=maxnorm(3)))
net.add(MaxPooling2D())  # default size is (2,2)
net.add(Flatten())
net.add(Dense(300, activation='relu', kernel_constraint=maxnorm(3)))  # or 512
net.add(Dropout(0.3))
net.add(Dense(classes_count, activation='softmax'))


# visually checking dataset
def show_train_examples():
    n = 9
    plt.figure(figsize=(20, 10))
    for i in range(n):
        plt.subplot(330 + 1 + i)
        plt.imshow(train_x[i])
    plt.show()

# if __name__ == '__main__':
#     show_train_examples()
