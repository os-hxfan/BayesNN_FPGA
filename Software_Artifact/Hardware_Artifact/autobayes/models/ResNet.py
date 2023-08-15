from re import X
import numpy as np
import tensorflow as tf 
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model,load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda, MaxPool2D, GlobalAveragePooling2D

def residual_conv_block(filters, down_sample=False):
    def layer(input_tensor):

        res = input_tensor
        strides = [2, 1] if down_sample else [1, 1]
        x = Conv2D(filters, strides=strides[0],
                             kernel_size=(3, 3), padding="same", kernel_initializer="he_normal")(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, strides=strides[1],
                             kernel_size=(3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        
        if down_sample:
            res = Conv2D(
                filters, strides=2, kernel_size=(1, 1), kernel_initializer="he_normal", padding="same")(res)
            res = BatchNormalization()(res)

        x = Add()([x, res])
        x = Activation('relu')(x)
        return x

    return layer

def ResNet18(filters, include_top=False, dense_out=100):
    keras.backend.set_image_data_format('channels_last')
    img_input = keras.Input(shape=[32,32,3])

    x = Conv2D(filters, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)
    for res_block in [residual_conv_block(filters), 
                      residual_conv_block(filters), 
                      residual_conv_block(filters * 2, down_sample=True),  
                      residual_conv_block(filters * 2), 
                      residual_conv_block(filters * 4, down_sample=True), 
                      residual_conv_block(filters * 4), 
                      residual_conv_block(filters * 8, down_sample=True), 
                      residual_conv_block(filters * 8)]:
            x = res_block(x)
    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(dense_out, activation='softmax')(x)
    return Model(img_input, x) 

# def conv_block(planes, strides):
#     def layer(input_tensor):
#         x = Conv2D(planes, strides=strides, kernel_size=(3, 3), padding='same')(input_tensor)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = Conv2D(planes, strides=1, kernel_size=(3, 3), padding='same')(x)
#         x = BatchNormalization()(x)
#         shortcut = input_tensor
#         if strides != 1:
#             shortcut = Conv2D(planes, strides=strides, kernel_size=(1, 1))(shortcut)
#             shortcut = BatchNormalization()(shortcut)
#         x = Add()([x, shortcut])
#         x = Activation('relu')(x)
#         return x 
#     return layer 

# def _make_layer(planes, num_blocks, stride):
#     strides = [stride] + [1]*(num_blocks-1)
#     layers = []
#     for stride in strides:
#         layers.append(conv_block(planes, stride))
#     return layers
   
# def ResNet():
#     img_input = keras.Input(shape=[32,32,3])
#     x = Conv2D(24, (3, 3), strides=1)(img_input)
#     x = ZeroPadding2D((1, 1))(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     for res_block in [*_make_layer(24, 2, 1), 
#                       *_make_layer(28, 2, 2),
#                       *_make_layer(96, 2, 2),
#                       *_make_layer(192, 2, 2)]:
#             x = res_block(x)
#     x = GlobalAveragePooling2D()(x)
#     x = Flatten()(x)
#     x = Dense(192)(x)
#     return Model(img_input, x)  

def ResNet(filters, include_top=False, dense_out=100): 
  keras.backend.set_image_data_format('channels_last')
  return ResNet18(filters, include_top, dense_out)
  ResNet18, _ = Classifiers.get('resnet18')
  model = ResNet18(input_shape=(224,224,3), include_top=False, weights='imagenet')
  return model

def ResNet50():
  keras.backend.set_image_data_format('channels_last')
  
  resnet = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3), include_top=False)
  
  # for layer in resnet.layers:
  #   layer.trainable = False 
  
  # input = keras.Input(shape=(32, 32, 3))
  # x = resnet(input)
  # x = Flatten()(x)
  # x = BatchNormalization()(x)
  # x = Dense(256, activation="relu")(x)
  # x = BatchNormalization()(x)
  # x = Dense(128, activation="relu")(x)
  # x = BatchNormalization()(x)
  # x = Dense(64, activation="relu")(x)
  # x = BatchNormalization()(x)
  # output = Dense(10, activation="softmax")(x)
  
  return resnet
