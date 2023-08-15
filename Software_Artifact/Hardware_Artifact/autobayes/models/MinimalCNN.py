from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import Sequential

def MinimalCNN(filters, include_top, dense_out):
  model = Sequential([
    Conv2D(8, (5,5), input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=6),
    Flatten(),
    Dense(10, activation='softmax'),
  ])
  return model 