#!/usr/bin/env python3
import sys
sys.path.append(sys.path[0] + '/..')
sys.path.append(sys.path[0] + '/../converter/keras')
import keras 
from converter.keras.MCDropout import MCDropout
from models.LeNet import LeNet 
from models.ResNet import ResNet18
from models.VGG import VGG11
import sys
import timeit 

def time_cost(network, name, *args):
  
  keras_conversion_time = 0
  hls_conversion_time = 0
  f = open('/dev/null', 'w')
  loop_num = 100
  for i in range(loop_num):
    print('Loop ' + str(i))
    
    model = network(*args)
    old_output = sys.stdout
    sys.stdout = f
    
    start = timeit.default_timer()
    # convert to Bayesian Keras model
    mc_model = MCDropout(model, strategy='full')
    end = timeit.default_timer()
    keras_conversion_time += end - start
  
    mc_model.addHlsConfig(OutputDir='timing/'+name, IOType='io_stream')
    start = timeit.default_timer()
    # convert to HLS model
    mc_model.compileHlsModel() 
    end = timeit.default_timer()
    hls_conversion_time += end - start
    
    sys.stdout = old_output
  
  f.close()
  keras_conversion_time /= loop_num
  hls_conversion_time /= loop_num
  print('Keras conversion time: ', keras_conversion_time)
  print('HLS conversion time: ', hls_conversion_time)
  print('Total conversion time: ', keras_conversion_time + hls_conversion_time)
  with open('timing/' + name + '.txt', 'w') as f:
    f.write('Keras conversion time: ' + str(keras_conversion_time) + '\n')
    f.write('HLS conversion time: ' + str(hls_conversion_time) + '\n')
    f.write('Total conversion time: ' + str(keras_conversion_time + hls_conversion_time))
    f.close()

keras.backend.set_image_data_format('channels_last')

network_map = {'ResNet18' : (ResNet18, 5),
               'LeNet' : (LeNet, None),
               'VGG11' : (VGG11, [16, 16, 10])}
network = network_map[sys.argv[1]]
time_cost(network[0], sys.argv[1], 16, True, network[1])