#!/usr/bin/env python3
import sys
sys.path.append(sys.path[0] + '/..')
sys.path.append(sys.path[0] + '/../converter/keras')
from converter.keras.Masksembles import MasksemblesModel
import keras 
from models.LeNet import LeNet 
from models.ResNet import ResNet18
from models.VGG import VGG11
import os
import sys
from contextlib import redirect_stdout

keras.backend.set_image_data_format('channels_last')

network_map = {'ResNet18' : (ResNet18, 5),
               'LeNet' : (LeNet, None),
               'VGG11' : (VGG11, [16, 16, 10])}

network = network_map[sys.argv[2]]
n = int(sys.argv[1])
model = network[0](16, True, network[1])
folder = 'diff_scales/' + sys.argv[2] + '-' + str(n)  
model = MasksemblesModel(model, num_masks=4, scale=n, strategy='full') 
os.mkdir(folder)
with open(folder + '/summary', 'w') as f:
    with redirect_stdout(f):
        model.model.build(input_shape=[32,32,3])
        model.model.summary()
    f.close()
hls_config, cfg = model.addHlsConfig(granularity='type')

hls_config['Model']['Precision'] = 'ap_fixed<8,8>'
hls_config['Model']['ReuseFactor'] = 90000000
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['BramFactor'] = 50000
hls_config['Model']['MergeFactor'] = 1
# Create an entry for each layer, here you can for instance change the strategy for a layer to 'resource' 
# or increase the reuse factor individually for large layers.
for Layer in hls_config['LayerType'].keys():
  if not Layer.startswith('Dense'):
    hls_config['LayerType'][Layer]['Compression'] = True
  hls_config['LayerType'][Layer]['Strategy'] = 'Resource'
  hls_config['LayerType'][Layer]['ReuseFactor'] = 90000000
  hls_config['LayerType'][Layer]['Precision'] = 'ap_fixed<8,8>'

cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
cfg['OutputDir']  = folder
cfg['XilinxPart'] = 'xcvu13p-flga2577-2-e'
cfg['Part'] = 'xcvu13p-flga2577-2-e'
cfg['ClockPeriod'] = 5
model.compileHlsModel() 
model.buildHlsModel(csim=False)


