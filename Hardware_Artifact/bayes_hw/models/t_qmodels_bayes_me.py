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
from converter.keras.MCDropout import BayesianDropout
from converter.keras.Masksembles import Masksembles
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
import math
from model_utils import Insert_Bayesian_Layer, Bayesian_Layer


##############################  Temporal_QLeNet_MCME   ####################################
######################   We adopt last-layer dropout as an example ########### 
def T_Qlenet_bayes_me(args, num_layers, name="t_qlenet"):
    input_shape=(28,28,1)
    input = Input(shape=input_shape)
    x = input
    mc_samples = args.mc_samples
    num_nonbayes_layer = num_layers - args.num_bayes_layer - 1
    # Lenet
    # Convolutional layer  
    x = QConv2D(filters=20, kernel_size=(5,5), padding = "same",
        kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), 
        bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), name="conv2d_1")(x)
    x = QActivation(activation=quantized_relu(args.quant_tbit), name='relu1')(x)
    # Max-pooing layer with pooling window size is 2x2
    x = MaxPool2D(pool_size=(2,2), strides=2)(x)
    
    mc_xs = [] # List to get all the mc samples from different exits
    ############## Second Exit ##############
    out1 = QConv2D(filters=20, kernel_size=(5,5), padding="same",
      kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), 
      bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), strides=(7,7), name="conv2d_2_2nd_exit")(x)
    out1 = QActivation(activation=quantized_relu(args.quant_tbit), name='relu2_2nd_exit')(out1)
 
    out1 = Flatten()(out1)
 
    out1 = QDense(100, kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), name="fc_1_2nd_exit")(out1)
    out1 = QActivation(activation=quantized_relu(args.quant_tbit), name='relu3_2nd_exit')(out1) 
    # Declare layers after first dropout
    bayes_2nd_exit = Bayesian_Layer(args)
    fc_2nd_exit = QDense(10, kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), 
        bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), name="fc_2nd_exit")
    softmax_2nd_exit = Activation(activation='softmax', name='softmax_2nd_exit')

    # Due to the "Cloning cannot exceed 2" limitation, we use Reshape to do Intermediate Layer Caching 
    '''
    ic_2nd = QActivation(activation=quantized_relu(args.quant_tbit), name='clone_2nd_exit')(out1)
    ic_2nd = [ic_2nd for i in range(2)] # Copy
    for j in range(math.ceil(math.log(mc_samples-1,2))-1):
        for i in range(len(ic_2nd)):
            #ic_2nd[i] = Reshape((-1,), name='reshape_%i'%(i))(ic_2nd[i])
            ic_2nd[i] = QActivation(activation=quantized_relu(args.quant_tbit), name='clone_2nd_exit_%i_%i'%(j,i))(ic_2nd[i])
        ic_2nd = ic_2nd * 2
    '''
    ################### Original Computation ###############
    orig_exit2 = bayes_2nd_exit(out1)
    orig_exit2 = fc_2nd_exit(orig_exit2)
    orig_exit2 = softmax_2nd_exit(orig_exit2)

    # Temporal Mapping Optimization for Bayesian Samples
    for i in range(mc_samples-1):
      # mc_x = bayes_2nd_exit(ic_2nd[i])
      mc_x = bayes_2nd_exit(out1)
      mc_x = fc_2nd_exit(mc_x)
      mc_x = softmax_2nd_exit(mc_x)
      mc_xs = mc_xs + [mc_x]


    ############## Main Exit ##############
    # Convolutional layer 
    x = QConv2D(filters=20, kernel_size=(5,5), padding="same",
      kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), 
      bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), name="conv2d_2")(x)
    x = QActivation(activation=quantized_relu(args.quant_tbit), name='relu2')(x)
    # Max-pooling layer 
    x = MaxPool2D(pool_size=(7,7), strides=7)(x)

    # Flatten layer 
    x = Flatten()(x)

    # The first fully connected layer 
    x = QDense(100, kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), name="fc_1")(x)
    x = QActivation(activation=quantized_relu(args.quant_tbit), name='relu3')(x)
    
    # Declare layers after first dropout
    bayes_1st_exit = Bayesian_Layer(args)
    fc_1st_exit = QDense(10, kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), 
        bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), name="fc_exit_1st")
    softmax_1st_exit = Activation(activation='softmax', name='softmax_1st_exit')

    # Due to the "Cloning cannot exceed 2" limitation, we use Reshape to do Intermediate Layer Caching 
    '''
    ic_1st = QActivation(activation=quantized_relu(args.quant_tbit), name='clone_exit_1st')(out1)
    ic_1st = [ic_1st for i in range(2)] # Copy
    for j in range(math.ceil(math.log(mc_samples-1,2))-1):
        for i in range(len(ic_1st)):
            #ic_1st[i] = Reshape((-1,), name='reshape_%i'%(i))(ic_1st[i])
            ic_1st[i] = QActivation(activation=quantized_relu(args.quant_tbit), name='clone_exit_1st_%i_%i'%(j,i))(ic_1st[i])
        ic_1st = ic_1st * 2
    '''
    ################### Original Computation ###############
    orig_exit1 = bayes_1st_exit(x)
    orig_exit1 = fc_1st_exit(orig_exit1)
    orig_exit1 = softmax_1st_exit(orig_exit1)
    # Temporal Mapping Optimization for Bayesian Samples
    for i in range(mc_samples - 1):
      #mc_x = bayes_1st_exit(ic_1st[i])
      mc_x = bayes_1st_exit(x)
      mc_x = fc_1st_exit(mc_x)
      mc_x = softmax_1st_exit(mc_x)
      mc_xs = mc_xs + [mc_x]

    model = Model(input, [orig_exit1, orig_exit2], name=name)
    model.compile(optimizer=Adam(lr = args.lr, decay=0.0001), loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy', 'accuracy'])

    bayes_me_model = Model(input, [orig_exit1, orig_exit2] + mc_xs, name=name+'_bayes_me')
    bayes_me_model.compile(optimizer=SGD(lr = args.lr), loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'])
    
    return model, bayes_me_model

##############################  QResNet   ####################################
# Get from https://github.com/jerett/Keras-CIFAR10
def conv2d_bn(x, filters, kernel_size, block_num, args, j=0, weight_decay=.0, strides=(1, 1)):
    layer = QConv2DBatchnorm(filters=filters,
    # layer = QConv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
                   bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
                   padding='same',
                   #use_bias=False,
                   name='fused_convbn_{}_{}'.format(block_num, j),
                   kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=True,
                   #kernel_regularizer=l2(weight_decay),
                   )(x)
    # layer = QBatchNormalization()(layer)
    return layer


def conv2d_bn_relu(x, filters, kernel_size, block_num, args,j=0, weight_decay=.0, strides=(1, 1)):
    layer = conv2d_bn(x, filters, kernel_size, block_num, args, j, weight_decay, strides)
    layer = QActivation(activation=quantized_relu(args.quant_tbit),name='conv_act_%i_%i'%(block_num, j))(layer)
    #layer = Activation('relu')(layer)
    return layer


def ResidualBlock(x, filters, kernel_size, block_num, weight_decay, args, downsample=True):
    j = 0
    if downsample:
        # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
        residual_x = conv2d_bn(x, filters, kernel_size=1, block_num=block_num, args=args, j = j, strides=2)
        j = j + 1
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              block_num = block_num,
                              j = j,
                              args=args, 
                              weight_decay=weight_decay,
                              strides=stride,
                              )
    j = j + 1
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         block_num = block_num,
                         j = j,
                         args=args, 
                         weight_decay=weight_decay,
                         strides=1,
                         )
    out = layers.add([residual_x, residual])
    out = QActivation(activation=quantized_relu(args.quant_tbit),name='conv_act_%i_%i'%(block_num, j))(out)
    return out

def T_QResNet18_bayes_me(classes, input_shape, args, weight_decay=1e-4, base_filters=64, num_bayes_loc=8, name="t_qresnet"):
    input = Input(shape=input_shape)
    x = input
    mc_samples = args.mc_samples
    num_nonbayes_layer = num_bayes_loc - args.num_bayes_layer - 1
    x = conv2d_bn_relu(x, filters=base_filters, kernel_size=(3, 3), block_num=0, weight_decay=weight_decay, args=args, strides=(1, 1))

    # # conv 2
    x = ResidualBlock(x, filters=base_filters, kernel_size=(3, 3), block_num=1, weight_decay=weight_decay, args=args, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    x = ResidualBlock(x, filters=base_filters, kernel_size=(3, 3), block_num=2, weight_decay=weight_decay, args=args, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    # # conv 3
    x = ResidualBlock(x, filters=2*base_filters, kernel_size=(3, 3), block_num=3, weight_decay=weight_decay, args=args, downsample=True)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    x = ResidualBlock(x, filters=2*base_filters, kernel_size=(3, 3), block_num=4, weight_decay=weight_decay, args=args, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    # # conv 4
    x = ResidualBlock(x, filters=4*base_filters, kernel_size=(3, 3), block_num=5, weight_decay=weight_decay, args=args, downsample=True)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    x = ResidualBlock(x, filters=4*base_filters, kernel_size=(3, 3), block_num=6, weight_decay=weight_decay, args=args, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1


    # # conv 5
    x = ResidualBlock(x, filters=8*base_filters, kernel_size=(3, 3), block_num=7, weight_decay=weight_decay, args=args, downsample=True)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    ############## Second Exit ##############
    out1 = ResidualBlock(x, filters=8*base_filters, kernel_size=(3, 3), block_num=9, weight_decay=weight_decay, args=args, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    snd_exit_num_nonbayes_layer = num_nonbayes_layer
    if (snd_exit_num_nonbayes_layer < 0): out1 = Insert_Bayesian_Layer(args, out1)
    snd_exit_num_nonbayes_layer -= 1

    out1 = AveragePooling2D(pool_size=(4, 4), padding='valid')(out1)
    out1 = Flatten()(out1)
    out1 = Dense(classes)(out1)
    out1 = Activation(activation='softmax', name='softmax_2nd_exit')(out1)


    ############## Main Exit ##############
    x = ResidualBlock(x, filters=8*base_filters, kernel_size=(3, 3), block_num=8, weight_decay=weight_decay, args=args, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(classes)(x)
    x = Activation(activation='softmax', name='softmax')(x)

    model = Model(input, [x, out1], name='ResNet18')
    model.compile(optimizer=SGD(lr=args.lr, momentum=0.9, nesterov=False), loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr = args.lr, decay=0.0001), loss=['categorical_crossentropy'], metrics=['accuracy'])

    bayes_me_model = Model(input, [x, out1], name=name+'_bayes_me')
    bayes_me_model.compile(optimizer=SGD(lr=args.lr, momentum=0.9, nesterov=False), loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'])   
    return model, bayes_me_model


##############################  Temporal_QVGG   ####################################

def T_QVGG11_bayes_me(args, num_bayes_loc=7, filters=64, dense_out=[512, 512, 100], name="t_qvgg"):
    input_shape=(32,32,3)
    input = Input(shape=input_shape)
    x = input
    mc_samples = args.mc_samples
    num_nonbayes_layer = num_bayes_loc - args.num_bayes_layer - 1

    x = QConv2DBatchnorm(filters=filters, input_shape=(32,32,3), kernel_size=(3,3), padding="same",
              kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=True,
              name='fused_convbn_1')(x)
    x = QActivation(activation=quantized_relu(args.quant_tbit), name='relu1')(x)
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    x = QConv2DBatchnorm(filters=filters * 2, kernel_size=(3,3), padding="same",
              kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=True,
              name='fused_convbn_2')(x)
    x = QActivation(activation=quantized_relu(args.quant_tbit), name='relu2')(x)
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    x = QConv2DBatchnorm(filters=filters * 4, kernel_size=(3,3), padding="same",
              kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=True,
              name='fused_convbn_3')(x)
    x = QActivation(activation=quantized_relu(args.quant_tbit), name='relu3')(x)

    x = QConv2DBatchnorm(filters=filters * 4, kernel_size=(3,3), padding="same",
              kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=True,
              name='fused_convbn_4')(x)
    x = QActivation(activation=quantized_relu(args.quant_tbit), name='relu4')(x)
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1
    
    x = QConv2DBatchnorm(filters=filters * 8, kernel_size=(3,3), padding="same",
              kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=True,
              name='fused_convbn_5')(x)
    x = QActivation(activation=quantized_relu(args.quant_tbit), name='relu5')(x)

    x = QConv2DBatchnorm(filters=filters * 8, kernel_size=(3,3), padding="same",
              kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=True,
              name='fused_convbn_6')(x)
    x = QActivation(activation=quantized_relu(args.quant_tbit), name='relu6')(x)
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    ############## Second Exit ##############
    out1 = QConv2DBatchnorm(filters=filters * 8, kernel_size=(3,3), padding="same",
              kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1), strides=(2,2),
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=True,
              name='fused_convbn_7_2nd_exit')(x)
    out1 = QActivation(activation=quantized_relu(args.quant_tbit), name='relu7_2nd_exit')(out1)

    out1 = Flatten()(out1)

    snd_exit_num_nonbayes_layer = num_nonbayes_layer
    if (snd_exit_num_nonbayes_layer < 0): out1 = Insert_Bayesian_Layer(args, out1)
    snd_exit_num_nonbayes_layer -= 1

    out1 = Dense(dense_out[2], name="fc_2_2nd_exit")(out1)

    out1 = Activation(activation='softmax', name='softmax_2nd_exit')(out1)

    ############## Main Exit ##############
    x = QConv2DBatchnorm(filters=filters * 8, kernel_size=(3,3), padding="same",
              kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=True,
              name='fused_convbn_7')(x)
    x = QActivation(activation=quantized_relu(args.quant_tbit), name='relu7')(x)


    x = QConv2DBatchnorm(filters=filters * 8, kernel_size=(3,3), padding="same",
              kernel_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              bias_quantizer=quantized_bits(args.quant_tbit, args.quant_ibit, alpha=1),
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=True,
              name='fused_convbn_8')(x)
    x = QActivation(activation=quantized_relu(args.quant_tbit), name='relu8')(x)
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    x = Flatten()(x)

    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    ####### We keep last few linear/dense layer as full precision ########
    x = Dense(dense_out[0], activation='relu')(x)

    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    x = Dense(dense_out[1], activation='relu')(x)

    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(args, x)
    num_nonbayes_layer -= 1

    x = Dense(dense_out[2], name="fc_2")(x)

    x = Activation(activation='softmax', name='softmax')(x)

    optimizer = Adam(lr=args.lr, amsgrad=True)
    model = Model(input, [x, out1], name=name)
    model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'])
    bayes_me_model = Model(input, [x, out1], name=name+"_bayes_me")
    bayes_me_model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'])

    return model, bayes_me_model

