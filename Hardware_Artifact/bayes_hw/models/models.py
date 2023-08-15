from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import *
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras.utils import to_categorical
from qkeras import *
from tensorflow.keras.optimizers import Adam, SGD
from converter.keras.MCDropout import BayesianDropout
from keras.regularizers import l2
from keras import layers
import os
import argparse 
import numpy as np
from tensorflow.keras import activations

from re import X
import numpy as np
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model,load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda, MaxPool2D, GlobalAveragePooling2D
from model_utils import *

##############################  LeNet   ####################################
def lenet(args, num_layers):
    model=Sequential()
    num_nonbayes_layer = num_layers - args.num_bayes_layer - 1
    # Lenet
    # Convolutional layer  
    model.add(Conv2D(filters=20, kernel_size=(5,5), input_shape=(28,28,1), padding = "same", name="conv2d_1"))
    model.add(Activation(activations.relu, name='relu1'))
    # Max-pooing layer with pooling window size is 2x2
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    
    # MC dropout
    if (num_nonbayes_layer < 0): model.add(Get_Bayesian_Layer(args))
    num_nonbayes_layer -= 1

    # Convolutional layer 
    model.add(Conv2D(filters=20, kernel_size=(5,5), padding="same", name="conv2d_2"))
    model.add(Activation(activations.relu, name='relu2'))
    # Max-pooling layer 
    model.add(MaxPool2D(pool_size=(7,7), strides=7))

    # Flatten layer 
    model.add(Flatten())

    # MC dropout
    if (num_nonbayes_layer < 0): model.add(Get_Bayesian_Layer(args))
    num_nonbayes_layer -= 1

    # The first fully connected layer 
    model.add(Dense(100, name="fc_1"))
    model.add(Activation(activations.relu, name='relu3'))
    # The output layer  

    # MC dropout
    if (num_nonbayes_layer < 0): model.add(Get_Bayesian_Layer(args))
    num_nonbayes_layer -= 1

    model.add(Dense(10, name="fc_2"))
    model.add(Activation(activation='softmax', name='softmax'))
    model.compile(optimizer=SGD(lr = args.lr), loss=['categorical_crossentropy'], metrics=['accuracy']) 
    return model

##############################  ResNet   ####################################
# Get from https://github.com/jerett/Keras-CIFAR10
def conv2d_bn(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay)
                   )(x)
    layer = BatchNormalization()(layer)
    return layer


def conv2d_bn_relu(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = conv2d_bn(x, filters, kernel_size, weight_decay, strides)
    layer = Activation('relu')(layer)
    return layer


def ResidualBlock(x, filters, kernel_size, weight_decay, downsample=True):
    if downsample:
        # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
        residual_x = conv2d_bn(x, filters, kernel_size=1, strides=2)
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              strides=stride,
                              )
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         weight_decay=weight_decay,
                         strides=1,
                         )
    out = layers.add([residual_x, residual])
    out = Activation('relu')(out)
    return out

def ResNet18(classes, input_shape, args, num_bayes_loc=8, weight_decay=1e-4, base_filters=64):
    input = Input(shape=input_shape)
    x = input
    mc_samples = args.mc_samples
    num_nonbayes_layer = num_bayes_loc - args.num_bayes_layer - 1
    # x = conv2d_bn_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
    x = conv2d_bn_relu(x, filters=base_filters, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))

    # # conv 2
    x = ResidualBlock(x, filters=base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    x = ResidualBlock(x, filters=base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    # # conv 3
    x = ResidualBlock(x, filters=2*base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    x = ResidualBlock(x, filters=2*base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    # # conv 4
    x = ResidualBlock(x, filters=4*base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    x = ResidualBlock(x, filters=4*base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    # # conv 5
    x = ResidualBlock(x, filters=8*base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    x = ResidualBlock(x, filters=8*base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1
    
    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input, x, name='ResNet18')
    model.compile(optimizer=SGD(lr=args.lr, momentum=0.9, nesterov=False), loss=['categorical_crossentropy'], metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=args.lr, amsgrad=True), loss=['categorical_crossentropy'], metrics=['accuracy'])
    return model

def ResNetForCIFAR10(classes, name, input_shape, block_layers_num, weight_decay, lr):
    input = Input(shape=input_shape)
    base_filters = 16
    x = input
    x = conv2d_bn_relu(x, filters=base_filters, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))

    # # conv 2
    for i in range(block_layers_num):
        x = ResidualBlock(x, filters=base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 3
    x = ResidualBlock(x, filters=base_filters*2, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    for i in range(block_layers_num - 1):
        x = ResidualBlock(x, filters=base_filters*2, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 4
    x = ResidualBlock(x, filters=base_filters*4, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    for i in range(block_layers_num - 1):
        x = ResidualBlock(x, filters=base_filters*4, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    #x = AveragePooling2D(pool_size=(8, 8), padding='valid')(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input, x, name=name)
    model.compile(optimizer=SGD(lr=lr, momentum=0.9, nesterov=False), loss=['categorical_crossentropy'], metrics=['accuracy'])
    return model

def ResNet20ForCIFAR10(classes, input_shape, weight_decay, lr):
    return ResNetForCIFAR10(classes, 'resnet20', input_shape, 3, weight_decay,lr)


##############################  VGG   ####################################

def VGG11(args, num_bayes_loc=7, filters=64, dense_out=[512, 512, 100]):
  mc_samples = args.mc_samples
  num_nonbayes_layer = num_bayes_loc - args.num_bayes_layer - 1
  model = Sequential()
  model.add(Conv2D(input_shape=(32,32,3),filters=filters,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
  if (num_nonbayes_layer < 0): model.add(Get_Bayesian_Layer(args))
  num_nonbayes_layer -= 1

  model.add(Conv2D(filters=filters * 2,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
  if (num_nonbayes_layer < 0): model.add(Get_Bayesian_Layer(args))
  num_nonbayes_layer -= 1

  model.add(Conv2D(filters=filters * 4,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))

  model.add(Conv2D(filters=filters * 4,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
  if (num_nonbayes_layer < 0): model.add(Get_Bayesian_Layer(args))
  num_nonbayes_layer -= 1

  model.add(Conv2D(filters=filters * 8,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))

  model.add(Conv2D(filters=filters * 8,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
  if (num_nonbayes_layer < 0): model.add(Get_Bayesian_Layer(args))
  num_nonbayes_layer -= 1

  model.add(Conv2D(filters=filters * 8,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))

  model.add(Conv2D(filters=filters * 8,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  model.add(Flatten())
  # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
  if (num_nonbayes_layer < 0): model.add(Get_Bayesian_Layer(args))
  num_nonbayes_layer -= 1

  model.add(Dense(dense_out[0], activation='relu'))
  # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
  if (num_nonbayes_layer < 0): model.add(Get_Bayesian_Layer(args))
  num_nonbayes_layer -= 1

  model.add(Dense(dense_out[1], activation='relu'))
  # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
  if (num_nonbayes_layer < 0): model.add(Get_Bayesian_Layer(args))
  num_nonbayes_layer -= 1
  
  model.add(Dense(dense_out[2], activation='softmax'))
  optimizer = Adam(lr=args.lr, amsgrad=True)
  model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                 metrics=['accuracy'])
  return model 