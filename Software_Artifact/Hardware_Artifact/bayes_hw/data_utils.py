import keras
from keras.datasets import cifar10, mnist
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10

class CIFAR10Data(object):
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        print('CIFAR10 Training data shape:', self.x_train.shape)
        print('CIFAR10 Training label shape', self.y_train.shape)
        print('CIFAR10 Test data shape', self.x_test.shape)
        print('CIFAR10 Test label shape', self.y_test.shape)

    def get_stretch_data(self, subtract_mean=True):
        """
        reshape X each image to row vector, and transform Y to one_hot label.
        :param subtract_mean:Indicate whether subtract mean image.
        :return: x_train, one_hot_y_train, x_test, one_hot_y_test
        """
        num_classes = len(self.classes)
        # x_train = np.reshape(self.x_train, (self.x_train.shape[0], -1)).astype('float64')
        x_train = np.reshape(self.x_train, (self.x_train.shape[0], -1)).astype('float32')
        y_train = keras.utils.to_categorical(self.y_train, num_classes)

        # x_test = np.reshape(self.x_test, (self.x_test.shape[0], -1)).astype('float64')
        x_test = np.reshape(self.x_test, (self.x_test.shape[0], -1)).astype('float32')
        y_test = keras.utils.to_categorical(self.y_test, num_classes)

        if subtract_mean:
            mean_image = np.mean(x_train, axis=0).astype('uint8')
            x_train -= mean_image
            x_test -= mean_image
            # print(x_mean[:10])
            # plt.figure(figsize=(4, 4))
            # plt.imshow(x_mean.reshape((32, 32, 3)))
            # plt.show()

        return x_train, y_train, x_test, y_test
    def get_data(self, subtract_mean=True, output_shape=None):
        """
        The data is not reshaped, keep 3 channel.
        :param subtract_mean:Indicate whether subtract mean image.
        :param output_shape:Indicate whether resize image
        :return: x_train, one_hot_y_train, x_test, one_hot_y_test
        """
        num_classes = len(self.classes)
        x_train = self.x_train
        x_test = self.x_test
        # if output_shape:resize
        #     x_train = np.array([cv2.resize(img, output_shape) for img in self.x_train])
        #     x_test = np.array([cv2.(img, output_shape) for img in self.x_test])

        x_train = x_train.astype('float32')
        y_train = keras.utils.to_categorical(self.y_train, num_classes)

        x_test = x_test.astype('float32')
        y_test = keras.utils.to_categorical(self.y_test, num_classes)

        if subtract_mean:
            mean_image = np.mean(x_train, axis=0)
            x_train -= mean_image
            x_test -= mean_image
        return x_train, y_train, x_test, y_test

SVHN_mean = tuple([x / 255 for x in [129.3, 124.1, 112.4]])
SVHN_std = tuple([x / 255 for x in [68.2, 65.4, 70.4]])
MNIST_mean = (0,)
MNIST_std = (1,)
CIFAR10_mean = (0.4914, 0.4822, 0.4465)
CIFAR10_std = (0.2023, 0.1994, 0.2010)
# generate random noise dataset 
def random_noise_data(dataset):
    if dataset == "mnist":
            # generate random noise test dataset with mean MNIST_mean and std MNIST_std
            x_test = np.random.normal(MNIST_mean, MNIST_std, [10000, 28, 28, 1])
            x_test = x_test.astype('float32')
            return x_test
    elif dataset == "cifar10":
            # generate random noise test dataset with mean CIFAR10_mean and std CIFAR10_std
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_test = np.random.normal(CIFAR10_mean, CIFAR10_std, x_test.shape)
            x_test = x_test.astype('float32')
            return x_test
    elif dataset ==  "svhn":
            # generate random noise test dataset with mean SVHN_mean and std SVHN_std
            x_test = np.random.normal(SVHN_mean, SVHN_std, [10000, 32, 32, 3])
            x_test = x_test.astype('float32')
            return x_test

            
