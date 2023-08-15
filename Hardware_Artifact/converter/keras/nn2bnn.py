import keras 
from keras import Model
import json
import hls4ml
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf

def _default_strategy(model, bayesian_layer, **kwargs):
  r"""It inserts Bayesian dropout layer before the last dense/conv layer.
      n is the number of layers to be inserted.
      The order of insertion is backward.
  Args:
      model (_type_): model to be converted
  """
  n = kwargs['num'] if 'num' in kwargs else 1
  conv_list = kwargs['conv_list'] if 'conv_list' in kwargs else {keras.layers.Conv2D}
  supported_layers = {}
  for i, layer in enumerate(model.layers):
    if type(layer) in conv_list or isinstance(layer, keras.layers.Dense):
      supported_layers[i-1] = bayesian_layer
  count = max(0, len(supported_layers) - n)
  for key in list(supported_layers):
    if count <= 0:
      break 
    del supported_layers[key]
    count -= 1
  return supported_layers

def _last_strategy(model, bayesian_layer, **kwargs):
  r"""It inserts Bayesian dropout layer before the first dense layer after the last Conv layer.
      n is the number of layers to be inserted.
      The order of insertion is backward.
  Args:
      model (_type_): model to be converted
  """
  n = kwargs['num'] if 'num' in kwargs else 1
  conv_list = kwargs['conv_list'] if 'conv_list' in kwargs else {keras.layers.Conv2D}
  supported_layers = {}
  after_conv = False
  last_conv = -1
  count = -1
  for i, layer in enumerate(model.layers):
    if type(layer) in conv_list:
      last_conv = i
      after_conv = True 
    if (isinstance(layer, keras.layers.Dense) and after_conv):
      count = i - 1
      after_conv = False 
  if (last_conv < 0):
    return supported_layers
  if (count < 0):
    count = last_conv
  while (n > 0 and count >= 0):
    supported_layers[count] = bayesian_layer
    count -= 1
    n -= 1
  return supported_layers

def _full_strategy(model, bayesian_layer, **kwargs):
  r"""It inserts Bayesian dropout layer after every layer of the model.
  Args:
      model (_type_): model to be converted
  """
  conv_list = kwargs['conv_list'] if 'conv_list' in kwargs else {keras.layers.Conv2D}
  supported_layers = {}
  for i, layer in enumerate(model.layers):
    if isinstance(layer, keras.engine.input_layer.InputLayer):
      continue 
    if type(layer) in conv_list or isinstance(layer, keras.layers.Dense):
      supported_layers[i-1] = bayesian_layer
  return supported_layers

def _convert_model(model, type, supported_layers, p=0.5, seed=0, n = 4, scale=1, input=None):
    r"""
    Args:
        model (_type_): model to be converted
        p (float, optional): dropout probability. Defaults to 0.5.
        input (_type_, optional): A dummy input to the model. Defaults to None.
        is_functional (bool, optional): whether the model is contruacted using functional api. Defaults to False.

    Returns:
        keras.Model: Bayesian model using Monte Carlo Dropout
    """
    model_arch = json.loads(model.to_json())
    is_functional = model_arch['class_name'] == 'Functional'
    if model_arch['class_name'] == 'Model':
      raise Exception('The subclassed model conversion is not supported yet! See {}' % model.name)

    
    if input is None and not hasattr(model, 'input'):
      raise AttributeError("Is the passed-in model created by subclassing? Please specify the input for the converted model!")
    input = model.input if input is None else input 
    x = input 

    dict = {'input_layers': {}, 'new_output': {}}
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in dict['input_layers']:
                dict['input_layers'].update(
                        {layer_name: [layer.name]})
            else:
                dict['input_layers'][layer_name].append(layer.name)

    dict['new_output'].update(
            {model.layers[0].name: input})
    output = []
    count = 0 
    for layer in model.layers: 
        if isinstance(layer, keras.engine.input_layer.InputLayer):
          count += 1
          continue 
        if is_functional:
          layer_input = [dict['new_output'][layer_aux] 
                  for layer_aux in dict['input_layers'][layer.name]]
          if len(layer_input) == 1:
              layer_input = layer_input[0]
        else:
          layer_input = x 
          
        if count in supported_layers:
            if type == 'BayesianDropout':
              bayesian_layer = supported_layers[count](p, seed)
            elif type == 'Masksembles':
              bayesian_layer = supported_layers[count](n, scale)
            else:
              raise Exception('The type {} is not supported yet!' % type)
            x = layer(layer_input)
            x = bayesian_layer(x)
        elif isinstance(layer, Model):
            x = _convert_model(layer, p, layer_input)(layer_input, supported_layers[count] or {})
        else: 
            x = layer(layer_input)

        if is_functional:
          dict['new_output'].update({layer.name: x})

          if (hasattr(model, 'output_names') and 
            isinstance(model.output_names, list) and 
            layer.name in model.output_names):
              output.append(x) 
        count += 1
    if not is_functional:
      output = x 
    return Model(inputs=input, outputs=output)

strategy_fn = {'default' : _default_strategy,
               'last' : _last_strategy,
               'full' : _full_strategy}

class HlsLayer(Model):
  r"""
   An abstract class for all bayesian layers that could be converted to HLS.
  """
  
  def __init__(self):
      super().__init__()
      
  def get_config(self):
    return super().get_config()

  @classmethod
  def from_config(cls, config):
    return cls(**config)
  
  def addHlsConfig(self, granularity='model', backend='Vivado', **kwargs):
    hls_config = hls4ml.utils.config_from_keras_model(self.model, granularity=granularity)
    cfg = hls4ml.converters.create_config(backend=backend)
    cfg['HLSConfig']  = hls_config
    cfg['KerasModel'] = self.model
    cfg['Bayes'] = True 
    for key, value in kwargs.items():
      cfg[str(key)] = value 
    self.config = cfg 
    return hls_config, cfg

  def compileHlsModel(self):
    if self.config is None:
      raise Exception('Must add config first!')
    self.hls_model = hls4ml.converters.keras_to_hls(self.config)
    self.hls_model.compile()
  
  def buildHlsModel(self, **kwargs):
    self.hls_model.build(**kwargs)

  def predict_hls(self, x_test, nSamples):
    if self.hls_model is None:
      raise Exception('Must add hls model first!')
    prediction = [self.hls_model.predict(x_test) for _ in range(nSamples)]
    return sum(prediction) / len(prediction)
  
  def evaluate_hls(self, x_test, y_test, proc_pred, nSamples):
    pred = proc_pred(self.predict_hls(x_test, nSamples))
    print("Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), pred)))
    
  def compare(self, model, x_test, plot_type = "dist_diff"):
    return hls4ml.model.profiling.compare(model, self.hls_model, x_test, plot_type)

