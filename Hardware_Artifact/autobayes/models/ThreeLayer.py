from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l1

def ThreeLayer():
  model = Sequential()
  model.add(Dense(64, input_shape=[16], name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
  model.add(Activation(activation='relu', name='relu1'))
  model.add(Dense(32, name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
  model.add(Activation(activation='relu', name='relu2'))
  model.add(Dense(32, name='fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
  model.add(Activation(activation='relu', name='relu3'))
  model.add(Dense(5, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
  model.add(Activation(activation='softmax', name='softmax'))
  return model