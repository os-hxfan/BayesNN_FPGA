from __future__ import print_function
from keras.metrics import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import mnist, cifar10
from sklearn.metrics import accuracy_score
import numpy as np 
import keras 
import tensorflow as tf 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
from keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from time import time
import json

def data_train(model: keras.Model, data, epochs=None, name=None, compile=True, loss_fn=None, opt='Adam', lr=0.001, callbacks=None):
  x_train, y_train, x_test, y_test = data  
  if compile:
    opt=Adam(learning_rate=lr)
    model.compile(loss=loss_fn, 
                  optimizer=opt, 
                  metrics=['accuracy']) 
  
  if epochs is not None:
    model.fit(x_train, y_train, batch_size=1024,
              epochs=epochs, validation_split=0.25, shuffle=True, callbacks=callbacks)

  if name is not None:
    print('Saving the model...')
    if hasattr(model, 'model'):
      model.model.save(name)
    else:
      model.save(name)
  
  # evaluate the model
  # _, acc = model.evaluate(x_test, y_test, batch_size=128, verbose = 1)
  # print('%.3f' % (acc * 100.0))
  # print("Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1))))

def mnist_data():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
  x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
  print(x_train.shape)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')

  x_train = x_train/255.0
  x_test = x_test/255.0

  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)
  return (x_train, y_train), (x_test, y_test)

def mnist_train(model: keras.Model, epochs=None, name=None, compile=True):
  (x_train, y_train), (x_test, y_test) = mnist_data()
  data_train(model=model, data=(x_train, y_train, x_test, y_test), epochs=epochs, name=name, compile=compile, loss_fn='categorical_crossentropy')
  
def cifar_data():  
  (x_train, y_train) , (x_test, y_test) = cifar10.load_data()
  x_train = tf.keras.applications.resnet50.preprocess_input(x_train.astype('float32'))
  x_test = tf.keras.applications.resnet50.preprocess_input(x_test.astype('float32'))
  x_train /= 255
  x_test /= 255
  y_train = to_categorical(y_train, 10)
  y_test = to_categorical(y_test, 10)
  return (x_train, y_train) , (x_test, y_test)
  
def cifar_train(model: keras.Model, epochs=None, name=None, compile=True):
  (x_train, y_train) , (x_test, y_test) = cifar_data()
  data_train(model, (x_train, y_train, x_test, y_test), epochs, name, compile, 'categorical_crossentropy', 'adam')

def jet_tagging_data():
  data = fetch_openml('hls4ml_lhc_jets_hlf')
  x, y = data['data'], data['target']
  le = LabelEncoder()
  y = le.fit_transform(y)
  y = to_categorical(y, 5)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test)
  x_test = np.ascontiguousarray(x_test)
  return (x_train, y_train), (x_test, y_test)

def jet_tagging_train(model: keras.Model, epochs=None, name=None, compile=True):
  (x_train, y_train), (x_test, y_test) = jet_tagging_data()
  data_train(model, (x_train, y_train, x_test, y_test), epochs, name, compile, 'categorical_crossentropy')

def getReports(indir):
    data_ = {}
    
    report_vsynth = Path('{}/vivado_synth.rpt'.format(indir))
    report_csynth = Path('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))
    
    if report_vsynth.is_file() and report_csynth.is_file():
        print('Found valid vsynth and synth in {}! Fetching numbers'.format(indir))
        
        # Get the resources from the logic synthesis report 
        with report_vsynth.open() as report:
            lines = np.array(report.readlines())
            data_['lut']     = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
            data_['ff']      = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
            data_['bram']    = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
            data_['dsp']     = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
            data_['lut_rel'] = float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[5])
            data_['ff_rel']  = float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[5])
            data_['bram_rel']= float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[5])
            data_['dsp_rel'] = float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[5])
        
        with report_csynth.open() as report:
            lines = np.array(report.readlines())
            lat_line = lines[np.argwhere(np.array(['Latency (cycles)' in line for line in lines])).flatten()[0] + 3]
            data_['latency_clks'] = int(lat_line.split('|')[2])
            data_['latency_mus']  = float(lat_line.split('|')[2])*5.0/1000.
            data_['latency_ii']   = int(lat_line.split('|')[6])
    
    return data_
      
