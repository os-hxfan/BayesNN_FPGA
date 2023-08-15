# Import necessary components to build LeNet
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.regularizers import l2

def AlexNet():
  model = Sequential([
      Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
      BatchNormalization(),
      MaxPooling2D(pool_size=(3,3), strides=(2,2)),
      Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
      BatchNormalization(),
      MaxPooling2D(pool_size=(3,3), strides=(2,2)),
      Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
      BatchNormalization(),
      Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
      BatchNormalization(),
      Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
      BatchNormalization(),
      MaxPooling2D(pool_size=(3,3), strides=(2,2)),
      Flatten(),
      Dense(4096, activation='relu'),
      Dropout(0.5),
      Dense(4096, activation='relu'),
      Dropout(0.5),
      Dense(10, activation='softmax')
  ])
  
  return model
