from keras.datasets import cifar10
import matplotlib.pyplot as plt

# importing needed model parameters
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils.np_utils import to_categorical


def load_data():
    # loading training and testing data
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    # changing type to float and normalizing data
    train_x = train_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255

    # performing one-hot encoding for classes - mapping class to integer (1 on the index of that integer)
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    return train_x, train_y, test_x, test_y


def define_net(classes_count):
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

    # defining gradient descent optimizer
    sgd_optimizer = SGD(momentum=0.8, decay=(0.01 / 25))  # momentum=0.9

    # compiling model
    net.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    net.summary()
    return net


def train_net(net, train_x, train_y, test_x, test_y):
    net.fit(train_x, train_y, batch_size=32, epochs=10, verbose='auto', validation_data=(test_x, test_y))
    _, accuracy = net.evaluate(test_x, test_y, batch_size=32)
    name = "net_model_{accuracy:f}.h5".format(accuracy=accuracy)
    net.save(name)


# visually checking dataset
def show_train_examples(train_x):
    n = 9
    plt.figure(figsize=(20, 10))
    for i in range(n):
        plt.subplot(330 + 1 + i)
        plt.imshow(train_x[i])
    plt.show()


if __name__ == '__main__':
    (train_image, train_class, test_image, test_class) = load_data()
    # getting total amount of classes
    count = train_class.shape[1]
    model = define_net(count)
    # show_train_examples(train_image)
    train_net(model, train_image, train_class, test_image, test_class)
