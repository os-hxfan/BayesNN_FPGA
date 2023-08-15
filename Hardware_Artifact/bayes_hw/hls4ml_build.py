#!/usr/bin/env python3
import sys
sys.path.append(sys.path[0] + '/..')
sys.path.append(sys.path[0] + '/../converter/keras')
sys.path.append(sys.path[0] + '/models')
import keras 
from converter.keras.MCDropout import MCDropout, BayesianDropout
from converter.keras.Masksembles import MasksemblesModel, Masksembles
import sys
from train_qkeras import get_model
from converter.keras.train import mnist_data
from keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import argparse 
import os
from tensorflow.keras.models import load_model
import numpy as np
import math
from model_utils import *

# conv: in * out * chnl * filter * scale
# fc: in * out * scale
resnet_scale = 8
vgg_scale = 16
opt_reuse_factors = {
"lenet" : {"conv2d_1" : 28*28*1*20*2,
        "conv2d_2" : 14*14*20*20*5*5,
        "fc_1" : 80*100*4},
"resnet" : {"fused_convbn_0_0" : 32*32*3*16*3*3*resnet_scale, #V3
        "fused_convbn_1_0" : 32*32*16*16*3*3*resnet_scale/2,
        "fused_convbn_1_1" : 32*32*16*16*3*3*resnet_scale/2,
        "fused_convbn_2_0" : 32*32*16*16*3*3*resnet_scale/2,
        "fused_convbn_2_1" : 32*32*16*16*3*3*resnet_scale/2,
        "fused_convbn_3_0" : 16*16*16*32*1*1*resnet_scale/2,
        "fused_convbn_3_1" : 16*16*16*32*3*3*resnet_scale/2,
        "fused_convbn_3_2" : 16*16*32*32*3*3*resnet_scale/2,
        "fused_convbn_4_0" : 16*16*32*32*3*3*resnet_scale/2,
        "fused_convbn_4_1" : 16*16*32*32*3*3*resnet_scale/2,
        "fused_convbn_5_0" : 8*8*32*64*1*1*resnet_scale,
        "fused_convbn_5_1" : 8*8*32*64*3*3*resnet_scale,
        "fused_convbn_5_2" : 8*8*64*64*3*3*resnet_scale,
        "fused_convbn_6_0" : 8*8*64*64*3*3*resnet_scale,
        "fused_convbn_6_1" : 8*8*64*64*3*3*resnet_scale,
        "fused_convbn_7_0" : 4*4*64*128*1*1*resnet_scale,
        "fused_convbn_7_1" : 4*4*64*128*3*3*resnet_scale,
        "fused_convbn_7_2" : 4*4*128*128*3*3*resnet_scale,
        "fused_convbn_8_0" : 4*4*128*128*3*3*resnet_scale,
        "fused_convbn_8_1" : 4*4*128*128*3*3*resnet_scale
        },
"vgg" : {"fused_convbn_1" : 32*32*3*16*3*3*vgg_scale, #V7
        "fused_convbn_2" : 16*16*16*32*3*3*vgg_scale,
        "fused_convbn_3" : 8*8*32*64*3*3*vgg_scale,
        "fused_convbn_4" : 8*8*64*64*3*3*vgg_scale,
        "fused_convbn_5" : 4*4*64*128*3*3*vgg_scale,
        "fused_convbn_6" : 4*4*128*128*3*3*vgg_scale,
        "fused_convbn_7" : 2*2*128*128*3*3*vgg_scale,
        "fused_convbn_8" : 2*2*128*128*3*3*vgg_scale}
}

def convert_build(args):
    co = {"BayesianDropout": BayesianDropout, "MCDropout": MCDropout, "Masksembles": Masksembles, "MasksemblesModel": MasksemblesModel}
    _add_supported_quantized_objects(co)
    model = load_model(args.load_model + '.h5', custom_objects=co)

    model.summary()
    import hls4ml
    #import plotting

    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['Activation'])
    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(rounding_mode='AP_RND')
    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(saturation_mode='AP_SAT')

    #First, the baseline model
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')

    # Set the precision and reuse factor for the full model

    hls_config['Model']['ReuseFactor'] = 1
    hls_config['Model']['Strategy'] = 'Latency'

    hls_config['Model']['BramFactor'] = 50000
    hls_config['Model']['MergeFactor'] = 1

    # Create an entry for each layer, here you can for instance change the strategy for a layer to 'resource'
    # or increase the reuse factor individually for large layers.
    # In this case, we designed the model to be small enough for a fully parallel implementation
    # so we use the latency strategy and reuse factor of 1 for all layers.
    opt_reuse_factor = opt_reuse_factors[args.model_name]
    for Layer in hls_config['LayerName'].keys():
        if (args.strategy == "latency") and ('mc' not in Layer):
            if (Layer in opt_reuse_factor):
                hls_config['LayerName'][Layer]['Strategy'] = 'Resource'
                hls_config['LayerName'][Layer]['ReuseFactor'] = int(math.ceil(opt_reuse_factor[Layer]/args.mem_limit))
                print (Layer, " uses optimized reuse factor ", hls_config['LayerName'][Layer]['ReuseFactor'])# = int(math.ceil(opt_reuse_factor[Layer]/args.mem_limit))
            else:
                hls_config['LayerName'][Layer]['Strategy'] = 'Resource' # V4
                hls_config['LayerName'][Layer]['ReuseFactor'] = 90000000
        else:
            hls_config['LayerName'][Layer]['Strategy'] = 'Resource' # Old Version
            hls_config['LayerName'][Layer]['ReuseFactor'] = 90000000

    cfg = hls4ml.converters.create_config(backend='Vivado')
    cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
    cfg['HLSConfig']  = hls_config
    cfg['KerasModel'] = model
    cfg['OutputDir']  = args.output_dir + '/'
    cfg['XilinxPart'] = 'xcku115-flvb2104-2-i'
    cfg['Part'] = 'xcku115-flvb2104-2-i'
    cfg['Bayes'] = True
    cfg['ClockPeriod'] = 5.5
    
    hls_model = hls4ml.converters.keras_to_hls(cfg)
    hls_model.compile()
    hls_model.build(csim=False, synth=True, vsynth=True, export=True)


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_model", default=None, type=str, help="Name of load model")
    parser.add_argument("--model_name", default="lenet", type=str, help="Optimization mode for MC sampling part", choices=["lenet", "vgg", "resnet"])
    parser.add_argument("--quant_ibit", default=0, type=int, help="The integer bits of quant")
    parser.add_argument("--num_bayes_layer", default=1, type=int, help="The number of Bayesian Layer")
    parser.add_argument("--output_dir", default='temp', type=str, help="Output directory")
    parser.add_argument("--strategy", default='resource', type=str, help="Stategy for implmenetation, latency or resource")
    parser.add_argument("--num_bins", default=10, type=int, help="The number of bins while calculating ECE")
    parser.add_argument("--num_mc_samples", default=1, type=int, help="The number of MC samples")
    parser.add_argument("--dropout_rate", default=0.2, type=float, help="The dropout rate")
    parser.add_argument("--num_masks", default=4, type=int, help="The number of masks")
    parser.add_argument("--scale", default=4, type=float, help="The scale")
    parser.add_argument("--mem_limit", default=4096, type=int, help="Mem limit for implementation")
    parser.add_argument("--dropout_type", default="mc", type=str, choices=["mc", "mask"], help="Dropout type, Monte-Carlo Dropout (mc) or Mask Ensumble (mask)")
    parser.add_argument("--is_me", default=0, type=int, help="Whether use multi-exit, 0 denote no use")
    parser.add_argument("--num_exits", default=2, type=int, help="The number of exits in multi-exit arch")

    args = parser.parse_args()

    convert_build(args)
