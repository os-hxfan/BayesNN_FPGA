from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np

co = {}
_add_supported_quantized_objects(co)
model = load_model('mnist_lenet.h5', custom_objects=co)

import hls4ml
#import plotting

#hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
#hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
#hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

#First, the baseline model
hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')

# Set the precision and reuse factor for the full model
#hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
#hls_config['Model']['ReuseFactor'] = 1

hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
hls_config['Model']['ReuseFactor'] = 90000000
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['BramFactor'] = 50000
hls_config['Model']['MergeFactor'] = 1

# Create an entry for each layer, here you can for instance change the strategy for a layer to 'resource'
# or increase the reuse factor individually for large layers.
# In this case, we designed the model to be small enough for a fully parallel implementation
# so we use the latency strategy and reuse factor of 1 for all layers.
for Layer in hls_config['LayerName'].keys():
    #hls_config['LayerName'][Layer]['Strategy'] = 'Latency'
    #hls_config['LayerName'][Layer]['ReuseFactor'] = 1
    hls_config['LayerName'][Layer]['Strategy'] = 'Resource'
    hls_config['LayerName'][Layer]['ReuseFactor'] = 90000000
#If you want best numerical performance for high-accuray models, while the default latency strategy is faster but numerically more unstable
hls_config['LayerName']['softmax']['Strategy'] = 'Stable'
#plotting.print_dict(hls_config)

cfg = hls4ml.converters.create_config(backend='Vivado')
cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
cfg['HLSConfig']  = hls_config
cfg['KerasModel'] = model
cfg['OutputDir']  = 'pruned_cnn/'
cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'

hls_model = hls4ml.converters.keras_to_hls(cfg)
hls_model.compile()

num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()

RESHAPED = 784

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

x_train /= 256.0
x_test /= 256.0
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_test_reduced = x_test[:2000]
y_test_reduced = y_test[:2000]

from sklearn.metrics import accuracy_score

y_predict        = model.predict(x_test_reduced)
accuracy_keras  = float(accuracy_score (np.argmax(y_test_reduced,axis=1), np.argmax(y_predict,axis=1)))
print("Accuracy Keras:  {}".format(accuracy_keras))
y_predict_hls4ml = hls_model.predict(np.ascontiguousarray(x_test_reduced))
accuracy_hls4ml = float(accuracy_score (np.argmax(y_test_reduced,axis=1), np.argmax(y_predict_hls4ml,axis=1)))
print("Accuracy hls4ml: {}".format(accuracy_hls4ml))
