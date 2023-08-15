import keras
from keras import Sequential
from keras.layers import Conv2D, Activation, Dropout, Flatten, Dense

def ReducedCNN():
  keras.backend.set_image_data_format('channels_last')
  model = Sequential()
  model.add(Conv2D(input_shape=(32,32,3), filters=32, 
                  use_bias=True, kernel_size=(3,3), activation='relu'))
  model.add(Dropout(0.1))
  model.add(Conv2D(filters=64, use_bias=False, kernel_size=(5,5), strides=2, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(10, activation="softmax"))
  return model