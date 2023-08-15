import torch
import torch.nn as nn 
import torch.nn.functional as F
from Dropouts import BayesianDropout, BayesianDropout2D, BayesianDropout3D
from test.ThreeLayerNet import ThreeLayerNet

class MCDropout(nn.Module):
  r"""
   It uses Morte Carlo Dropout to convert a 
   traditional neural network to a Bayesian neural network. The mathematical 
   proof is in this paper: https://arxiv.org/pdf/1506.02142.pdf. 
  """
  def __init__(self, model, nSamples=10, p=0.5):
    super(MCDropout, self).__init__()
    self.model = _convert_model(model, p)
    self.nSamples = nSamples
    self.p = p 
  
  def forward(self, x):
    # during training, it behaves like normal dropout models
    if (self.training):
      return self.model(x)
    # during inference, multiple samples are run and the average is returned
    else:
      print ("====================", pred[0].shape, pred[1].shape)
      pred = [self.model(x) for _ in range(self.nSamples)]
      return sum(pred) / len (pred)
  
  def extra_repr(self) -> str:
    return "nSamples: {}\nprobability: {}".format(self.nSamples, self.p)

def _convert_model(model, p):
  base_layers = {nn.Linear : BayesianDropout, 
                 nn.MaxPool1d : BayesianDropout,
                 nn.MaxPool2d : BayesianDropout,
                 nn.MaxPool3d : BayesianDropout,
                 nn.Conv1d : BayesianDropout, 
                 nn.Conv2d : BayesianDropout2D, 
                 nn.Conv3d : BayesianDropout3D}
  if type(model) in base_layers:
    return base_layers[type(model)](model, p)
  else:
    for name, layer in model.named_children():
      setattr(model, name, _convert_model(layer, p))
    return model

