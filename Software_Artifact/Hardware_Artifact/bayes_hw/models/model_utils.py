from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import *
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv2DBatchnorm
from qkeras.qpooling import QAveragePooling2D
from qkeras.qnormalization import QBatchNormalization
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras.utils import to_categorical
from qkeras import *
from tensorflow.keras.optimizers import Adam, SGD
from converter.keras.MCDropout import BayesianDropout, MCDropout
from converter.keras.Masksembles import Masksembles, MasksemblesModel
from keras.regularizers import l2, l1
from keras import layers
import os
import argparse 
import numpy as np

from re import X
import numpy as np
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model,load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda, MaxPool2D, GlobalAveragePooling2D, Reshape, Concatenate
import keras.backend as K


def Insert_Bayesian_Layer(args, x):
  if args.dropout_type == "mc": x = BayesianDropout(args.dropout_rate)(x)
  elif args.dropout_type == "mask": x = Masksembles(n=args.num_masks, scale=args.scale)(x)
  else: raise NotImplementedError("dropout type is not supportred")
  return x

def Bayesian_Layer(args):
  if args.dropout_type == "mc": bayes_layer = BayesianDropout(args.dropout_rate)
  elif args.dropout_type == "mask": bayes_layer = Masksembles(n=args.num_masks, scale=args.scale)
  else: raise NotImplementedError("dropout type is not supportred")
  return bayes_layer

def Get_Bayesian_Layer(args):
  if args.dropout_type == "mc": return BayesianDropout(args.dropout_rate)
  elif args.dropout_type == "mask": return Masksembles(n=args.num_masks, scale=args.scale)
  else: raise NotImplementedError("dropout type is not supportred")

# needed for inference
def Top_Level_Model(args, model):
  if args.dropout_type == "mc": 
      model = MCDropout(model, nSamples=args.mc_samples, p=args.dropout_rate, num=0)
  elif args.dropout_type == "mask":
      model = MasksemblesModel(model, num_masks=args.num_masks, scale=args.scale, num=0) 
  else:
      raise NotImplementedError("dropout type is not supportred")
  model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=args.lr), metrics=["accuracy"])
  return model 