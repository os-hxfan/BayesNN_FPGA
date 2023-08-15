
#!/usr/bin/env python3
import sys
sys.path.append(sys.path[0] + '/..')
sys.path.append(sys.path[0] + '/../converter/keras')
from converter.keras.MCDropout import MCDropout, BayesianDropout
from converter.keras.Masksembles import MasksemblesModel, Masksembles
import keras 
from models.LeNet import LeNet 
from models.ResNet import ResNet18
from models.VGG import VGG11
import os
import sys
from contextlib import redirect_stdout
from converter.keras.train import *
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, ZeroPadding2D

keras.backend.set_image_data_format('channels_last')

import numpy as np


def generate_masks_(m: int, n: int, s: float) -> np.ndarray:

    total_positions = int(m * s)
    masks = []

    for _ in range(n):
        new_vector = np.zeros([total_positions])
        idx = np.random.choice(range(total_positions), m, replace=False)
        new_vector[idx] = 1
        masks.append(new_vector)

    masks = np.array(masks)
    # drop useless positions
    masks = masks[:, ~np.all(masks == 0, axis=0)]
    return masks


def generate_masks(m: int, n: int, s: float) -> np.ndarray:

    masks = generate_masks_(m, n, s)
    # hardcoded formula for expected size, check reference
    expected_size = int(m * s * (1 - (1 - 1 / s) ** n))
    while masks.shape[1] != expected_size:
        masks = generate_masks_(m, n, s)
    return masks


def generation_wrapper(c: int, n: int, scale: float) -> np.ndarray:

    if c < 10:
        raise ValueError("Masksembles approach couldn't be used in such setups where "
                         f"number of channels is less then 10. Current value is (channels={c}). "
                         "Please increase number of features in your layer or remove this "
                         "particular instance of Masksembles from your architecture.")

    if scale > 6.:
        raise ValueError("Masksembles approach couldn't be used in such setups where "
                         f"scale parameter is larger then 6. Current value is (scale={scale}).")

    # inverse formula for number of active features in masks
    active_features = int(int(c) / (scale * (1 - (1 - 1 / scale) ** n)))

    # FIXME this piece searches for scale parameter value that generates
    #  proper number of features in masks, sometimes search is not accurate
    #  enough and masks.shape != c. Could fix it with binary search.
    masks = generate_masks(active_features, n, scale)
    for s in np.linspace(max(0.8 * scale, 1.0), 1.5 * scale, 300):
        if masks.shape[-1] >= c:
            break
        masks = generate_masks(active_features, n, s)
    new_upper_scale = s

    if masks.shape[-1] != c:
        for s in np.linspace(max(0.8 * scale, 1.0), new_upper_scale, 1000):
            if masks.shape[-1] >= c:
                break
            masks = generate_masks(active_features, n, s)

    if masks.shape[-1] != c:
        raise ValueError("generation_wrapper function failed to generate masks with "
                         "requested number of features. Please try to change scale parameter")

    return masks

class Masksembles2D(tf.keras.layers.Layer):

    def __init__(self, n: int, scale: float):
        super(Masksembles2D, self).__init__()

        self.n = n
        self.scale = scale

    def build(self, input_shape):
        channels = input_shape[-1]
        masks = generation_wrapper(channels, self.n, self.scale)
        self.masks = self.add_weight("masks",
                                     shape=masks.shape,
                                     trainable=False,
                                     dtype="float32")
        self.masks.assign(masks)

    def call(self, inputs, training=False):
        # inputs : [N, H, W, C]
        # masks : [M, C]
        x = tf.stack(tf.split(inputs, self.n))
        # x : [M, N // M, H, W, C]
        # masks : [M, 1, 1, 1, C]
        x = x * self.masks[:, tf.newaxis, tf.newaxis, tf.newaxis]
        x = tf.concat(tf.split(x, self.n), axis=1)
        return tf.squeeze(x, axis=0)


class Masksembles1D(tf.keras.layers.Layer):
    def __init__(self, n: int, scale: float):
        super(Masksembles1D, self).__init__()

        self.n = n
        self.scale = scale

    def build(self, input_shape):
        channels = input_shape[-1]
        masks = generation_wrapper(channels, self.n, self.scale)
        self.masks = self.add_weight("masks",
                                     shape=masks.shape,
                                     trainable=False,
                                     dtype="float32")
        self.masks.assign(masks)

    def call(self, inputs, training=False):
        x = tf.stack(tf.split(inputs, self.n))
        x = x * self.masks[:, tf.newaxis]
        x = tf.concat(tf.split(x, self.n), axis=1)
        return tf.squeeze(x, axis=0)


num_masks = 4
scale = 2
epochs = 5
model = LeNet(20, True, 10)
#model = keras.models.load_model('test/lenet')
mnist_train(model, epochs=epochs, name='test/lenet')

bayes_model = MasksemblesModel(LeNet(20, True, 10), num_masks=num_masks, scale=scale, num=3)
bayes_model.model.summary()
mnist_train(bayes_model, epochs=epochs, name='test/lenet-bayes')

original_bayes_model = keras.Sequential([  
  Conv2D(filters=20, kernel_size=(5,5), input_shape=(28,28,1)),
  ZeroPadding2D(padding=(2, 2)), 
  MaxPool2D(pool_size=(2,2), strides=2), 
  Masksembles2D(num_masks, scale), 
  Conv2D(filters=20, kernel_size=(5,5)),
  ZeroPadding2D(padding=(2, 2)),  
  MaxPool2D(pool_size=(7,7), strides=7), 
  Flatten(),
  Masksembles1D(num_masks, scale),
  Dense(100, activation='relu'),
  Masksembles1D(num_masks, scale), 
  Dense(10, activation='softmax')
])
mnist_train(original_bayes_model, epochs=epochs, name='test/original-lenet-bayes')

# Test
(x_train, y_train), (x_test, y_test) = mnist_data() 
x_test = x_test[:100]
y_test = y_test[:100]

print("Model Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1))))

print("Bayesian Model Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(bayes_model.predict(x_test), axis=1))))

inputs = np.tile(x_test, [num_masks, 1, 1, 1])
predictions = original_bayes_model.predict(inputs)
predictions = np.mean(np.split(predictions, num_masks), axis=0)
print("Original Bayesian Model Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))))



