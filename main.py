import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import os

(train_x, train_y), (test_x, test_y) = cifar10.load_data()
# checking dataset
n=9
plt.figure(figsize=(20, 10))
for i in range(n):
    plt.subplot(330+1+i)
    plt.imshow(train_x[i])
plt.show()
