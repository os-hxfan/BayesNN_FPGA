from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD
# metrics 
from keras.metrics import categorical_crossentropy
# optimization method
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.datasets import mnist
import keras 

def LeNet1():
  keras.backend.set_image_data_format('channels_last')
  
  model=Sequential()
  
  # Convolutional layer  
  model.add(Conv2D(filters=4, kernel_size=(5,5), input_shape=(28,28,1)))
  
  # Max-pooing layer with pooling window size is 2x2
  model.add(MaxPool2D(pool_size=(2,2), strides=2))
  
  # Convolutional layer 
  model.add(Conv2D(filters=8, kernel_size=(5,5)))
  
  # Max-pooling layer 
  model.add(MaxPool2D(pool_size=(2,2), strides=2))
  
  model.add(Flatten())
  
  model.add(Dense(10, activation='softmax'))
  
  return model

def LeNet5(include_top=False):
  keras.backend.set_image_data_format('channels_last')
  
  model=Sequential()
  
  # Convolutional layer  
  model.add(Conv2D(filters=20, kernel_size=(5,5), input_shape=(28,28,1)))
  
  # Padding
  model.add(ZeroPadding2D(padding=(2, 2)))
  
  # Max-pooing layer with pooling window size is 2x2
  model.add(MaxPool2D(pool_size=(2,2), strides=2))
  
  # Convolutional layer 
  model.add(Conv2D(filters=50, kernel_size=(5,5)))
  
  # Padding
  model.add(ZeroPadding2D(padding=(2, 2)))
  
  # Max-pooling layer 
  model.add(MaxPool2D(pool_size=(2,2), strides=2))
  
  if include_top:
    # Flatten layer 
    model.add(Flatten())
    
    # The first fully connected layer 
    model.add(Dense(500, activation='relu'))
    
    # The output layer  
    model.add(Dense(10, activation='softmax'))
  return model

def LeNet(filter, include_top, dense_out):
  keras.backend.set_image_data_format('channels_last')
  
  model=Sequential()
  
  # Convolutional layer  
  model.add(Conv2D(filters=20, kernel_size=(5,5), input_shape=(28,28,1)))
  
  # Padding
  model.add(ZeroPadding2D(padding=(2, 2)))
  
  # Max-pooing layer with pooling window size is 2x2
  model.add(MaxPool2D(pool_size=(2,2), strides=2))
  
  # Convolutional layer 
  model.add(Conv2D(filters=20, kernel_size=(5,5)))
  
  # Padding
  model.add(ZeroPadding2D(padding=(2, 2)))
  
  # Max-pooling layer 
  model.add(MaxPool2D(pool_size=(7,7), strides=7))
  
  # Flatten layer 
  model.add(Flatten())
  
  # The first fully connected layer 
  model.add(Dense(100, activation='relu'))
  
  # The output layer  
  model.add(Dense(10, activation='softmax'))
  return model