import numpy as np
import tensorflow as tf 
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model,load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda, MaxPool2D

def VGG11(filters=64, include_top=False, dense_out=[512, 512, 100]):
  model = Sequential()
  model.add(Conv2D(input_shape=(32,32,3),filters=filters,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  model.add(Conv2D(filters=filters * 2,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  model.add(Conv2D(filters=filters * 4,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(filters=filters * 4,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  model.add(Conv2D(filters=filters * 8,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(filters=filters * 8,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  model.add(Conv2D(filters=filters * 8,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(filters=filters * 8,kernel_size=(3,3),padding="same"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  
  if include_top:
    model.add(Flatten())
    model.add(Dense(dense_out[0], activation='relu'))
    model.add(Dense(dense_out[1], activation='relu'))
    model.add(Dense(dense_out[2], activation='softmax'))
  return model 

def VGG16(filters, include_top=False):
  model = Sequential()
  model.add(Conv2D(input_shape=(32,32,3),filters=filters,kernel_size=(3,3),padding="same", activation="relu"))
  model.add(Conv2D(filters=filters,kernel_size=(3,3),padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  model.add(Conv2D(filters=filters * 2, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=filters * 2, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  model.add(Conv2D(filters=filters * 4, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=filters * 4, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=filters * 4, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  model.add(Conv2D(filters=filters * 8, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=filters * 8, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=filters * 8, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  model.add(Conv2D(filters=filters * 8, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=filters * 8, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(Conv2D(filters=filters * 8, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  if include_top:
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=10, activation="softmax"))
  return model 
  model = tf.keras.applications.VGG16(
      input_shape=(32, 32, 3),
      include_top=False,
      weights="imagenet")
  return model 