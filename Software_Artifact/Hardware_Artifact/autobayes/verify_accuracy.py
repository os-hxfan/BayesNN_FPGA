
#!/usr/bin/env python3
import sys
sys.path.append(sys.path[0] + '/..')
sys.path.append(sys.path[0] + '/../converter/keras')
from converter.keras.MCDropout import MCDropout, BayesianDropout
from converter.keras.Masksembles import MasksemblesModel, Masksembles
import keras 
from models.LeNet import LeNet 
from models.ResNet import ResNet18
from models.VGG import VGG11
import os
import sys
from contextlib import redirect_stdout
from converter.keras.train import *
import hls4ml

keras.backend.set_image_data_format('channels_last')
hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['Activation'])
hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(rounding_mode='AP_RND')
hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(saturation_mode='AP_SAT')

num_masks = 4
epochs = 1
#model = LeNet(20, True, 10)
model = keras.models.load_model('test/lenet')
#mnist_train(model, epochs=epochs, name='test/lenet')

bayes_model = MasksemblesModel(LeNet(20, True, 10), num_masks=num_masks, scale=4, num=3)
#bayes_model = keras.models.load_model('test/lenet-bayes', custom_objects={'Masksembles': Masksembles, 'MasksemblesModel': MasksemblesModel})
bayes_model.model.summary()
mnist_train(bayes_model, epochs=epochs, name='test/lenet-bayes')

# Non-bayesian model to hls model 
# hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
# hls_config['Model']['ReuseFactor'] = 90000000
# hls_config['Model']['Strategy'] = 'Resource'
# hls_config['Model']['BramFactor'] = 50000
# hls_config['Model']['MergeFactor'] = 1
# for Layer in hls_config['LayerName'].keys():
#     hls_config['LayerName'][Layer]['Strategy'] = 'Resource'
#     hls_config['LayerName'][Layer]['ReuseFactor'] = 90000000
#     if 'softmax' in Layer:
#       hls_config['LayerName'][Layer]['Strategy'] = 'Stable'
# cfg = hls4ml.converters.create_config(backend='Vivado')
# cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
# cfg['HLSConfig']  = hls_config
# cfg['KerasModel'] = model
# cfg['OutputDir']  = 'diff_masksembles/test1'
# cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'
# hls_model = hls4ml.converters.keras_to_hls(cfg)
# hls_model.compile()

# Bayesian model to hls model
hls_config = hls4ml.utils.config_from_keras_model(bayes_model.model, granularity='name')
hls_config['Model']['ReuseFactor'] = 90000000
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['BramFactor'] = 50000
hls_config['Model']['MergeFactor'] = 1
for Layer in hls_config['LayerName'].keys():
    hls_config['LayerName'][Layer]['Strategy'] = 'Resource'
    hls_config['LayerName'][Layer]['ReuseFactor'] = 90000000
    if 'softmax' in Layer:
      hls_config['LayerName'][Layer]['Strategy'] = 'Stable'
cfg = hls4ml.converters.create_config(backend='Vivado')
cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
cfg['HLSConfig']  = hls_config
cfg['KerasModel'] = bayes_model.model
cfg['OutputDir']  = 'diff_masksembles/test2'
cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'
cfg['Bayes'] = True
bayes_hls_model = hls4ml.converters.keras_to_hls(cfg)
bayes_hls_model.compile()

# Test
(x_train, y_train), (x_test, y_test) = mnist_data() 
x_test = x_test[:10]
y_test = y_test[:10]

# print("Model Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1))))

pred = bayes_model.predict(x_test)
# print(pred)
print("Bayesian Model Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))))

# print("HLS Model Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(hls_model.predict(x_test), axis=1))))

preds = [bayes_hls_model.predict(x_test, mask_index=i) for i in range(num_masks)]
for i in range(num_masks):
  # print(preds[i])
  print("Bayesian HLS Model Accuracy(Model {}): {}".format(i, accuracy_score(np.argmax(y_test, axis=1), np.argmax(preds[i], axis=1))))
print("HLS Model Accuracy(Average): {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(sum(preds) / num_masks, axis=1))))



