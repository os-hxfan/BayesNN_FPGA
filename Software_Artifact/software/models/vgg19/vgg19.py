"""
https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/vgg.py
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from utils import dict_drop
from utils import Masksembles2D, Masksembles1D

def get_vgg_19(ensemble, network_hyperparams):
    if ensemble is None:
        # n_exits, out_dim
        model = VGG19(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type","dropout","dropout_exit", "dropout_p", "mask_type", "num_masks", "mask_scale"))

    elif ensemble == "early_exit":
        # n_exits, out_dim
        model = VGG19EarlyExit(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type","dropout","dropout_exit", "dropout_p", "mask_type", "num_masks", "mask_scale"))

    elif ensemble == "mc":
        # n_exits, out_dim, dropout, dropout_exit, dropout_p
        model = VGG19MC(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type"))
     
    elif ensemble == "mc_early_exit":
        # n_exits, out_dim, dropout, dropout_exit, dropout_p
        model =  VGG19MCEarlyExit(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type"))
    else:
        raise ValueError

    if network_hyperparams["image_size"] == 224 and network_hyperparams["out_dim"] == 7:
        model = load_imagenet_weights(model)
        check_weight_load(model)
    return model

def load_imagenet_weights(model):
    state_dict = torch.load("./MultiExit_BNNs/models/model_weights/vgg19_bn-c79401a0.pth")
    key_list = ['classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias']
    for key in key_list:
        del state_dict[key]
    state_dict = vgg_pytorch_to_new_vgg(state_dict)
    model.load_state_dict(state_dict, strict = False)
    return model

def check_weight_load(model):
    if type(model) == VGG19 or model.dropout is None or model.dropout == "block":
        if math.isclose(model.blocks[4][9].weight[0][0][0][0].item(), -0.0026779528707265854):
            print("Model Weights Loaded")
        else:
            print("Not Loaded Correctly!")
    elif model.dropout == "layer":
        if math.isclose(model.blocks[4][18].weight[0][0][0][0].item(), -0.0026779528707265854):
            print("Model Weights Loaded")
        else:
            print("Not Loaded Correctly!")
    return None

def vgg_pytorch_to_new_vgg(state_dict):
    block = 0
    layer = 0
    prev_feature_num = 0
    keys_list = list(state_dict.keys()).copy()
    for key in keys_list:
        key_split = key.split(".")
        if int(key_split[1]) - prev_feature_num == 3:
            block += 1
            layer = 0
        elif int(key_split[1]) - prev_feature_num > 0:
            layer += 1
        new_key = "blocks."+str(block)+"."+str(layer)+"."+key_split[-1]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
        prev_feature_num = int(key_split[1])
        if key_split[-1] == "running_var":
            layer += 1
    return state_dict
        
# Below VGG and VGG19 classes are mainly adapted from: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py 
# and https://github.com/Lornatang/pytorch-vgg19-cifar100/blob/master/vgg.py 
class VGG(nn.Module):

    def __init__(self, blocks, num_class=100, image_size = 32):
        super().__init__()
        self.blocks, self.non_sequentialized_blocks = blocks
        self.image_size = image_size
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = make_classifier(self.image_size,num_class,mask_type=None)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        output = x
        for block in self.blocks:
            output = block(output)
        if self.image_size == 224:
            output = self.avg_pool(output)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        self.intermediary_output_list = (output, [], 0, [])
        return [output]

def make_layers(cfg, batch_norm=False):
    blocks = nn.ModuleList()
    layers = nn.ModuleList()

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            blocks.append(layers)
            layers = nn.ModuleList()
            continue

        layers.append(nn.Conv2d(input_channel, l, kernel_size=3, padding=1))

        if batch_norm:
            layers.append(nn.BatchNorm2d(l))

        layers.append(nn.ReLU(inplace=True))
        input_channel = l
    sequential_blocks = nn.ModuleList()
    for block in range(len(blocks)):
        sequential_blocks.append(nn.Sequential(*blocks[block]))
    return (sequential_blocks, blocks)


def make_classifier(size, num_classes, mc_dropout_p=0, mask_type='mask', num_masks=4, mask_scale=4.0):
    if size == 224:
        if mc_dropout_p == 0:
            classifier = nn.ModuleList([nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, num_classes)])
        else:
            if mask_type == 'mc':
                classifier = nn.ModuleList([nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    MCDropout(p=mc_dropout_p),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    MCDropout(p=mc_dropout_p),
                    nn.Linear(4096, num_classes)])   
            else:    
                classifier = nn.ModuleList([nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    Masksembles1D(4096, num_masks, mask_scale),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    Masksembles1D(4096, num_masks, mask_scale),
                    nn.Linear(4096, num_classes)])        
    else:
        if mc_dropout_p == 0:
            classifier =  nn.ModuleList([nn.Linear(512, num_classes)])
        else:
            if mask_type == 'mc':
                classifier =  nn.ModuleList([MCDropout(p = mc_dropout_p),
                                nn.Linear(512, num_classes)])
            else:
                classifier =  nn.ModuleList([Masksembles1D(512, num_masks, mask_scale),
                                nn.Linear(512, num_classes)])
    return nn.Sequential(*classifier)


class VGG19(VGG):
    def __init__(self, n_exits = 1, out_dim = 100, *args,  **kwargs):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        super().__init__(make_layers(cfg, batch_norm=True), num_class=out_dim, *args,  **kwargs)
        self.n_exits = n_exits
        self.out_dim = out_dim
        self.init_weights()

class VGG19MC(VGG19):
    def __init__(self,dropout_exit = False, dropout = None, dropout_p = 0.5, n_exits = 1,
                    out_dim = 100, mask_type = "mc", num_masks = 4, mask_scale = 4.0, *args,  **kwargs):
        super().__init__(*args,  **kwargs)
        self.n_exits = n_exits
        self.out_dim = out_dim
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.dropout_exit = dropout_exit
        self.mask_type = mask_type
        self.num_masks = num_masks
        self.mask_scale = mask_scale
        dropout_all_blocks = False
        # Overwrite super classifier
        if self.dropout_exit:
            self.classifier = make_classifier(self.image_size,self.out_dim,self.dropout_p,self.mask_type,self.num_masks,self.mask_scale)
        # Load weights
        self.init_weights()
        if self.image_size == 224 and self.out_dim == 7:
            load_imagenet_weights(self)
            # Additional Linear layers between dropout, so can put dropout at end of all blocks
            dropout_all_blocks = True

        if self.dropout is not None:
            self.blocks = nn.ModuleList()
            for block in range(len(self.non_sequentialized_blocks)):
                if self.dropout == "block":
                    if not dropout_all_blocks and block == len(self.non_sequentialized_blocks)-1:
                        pass
                    else:
                        out_channels = self.non_sequentialized_blocks[block][-2].out_channels
                        dropout_layer = MCDropout(self.dropout_p) if (self.mask_type=="mc") else Masksembles1D(out_channels, self.num_masks, self.mask_scale) 
                        self.non_sequentialized_blocks[block].append(dropout_layer)
                elif self.dropout == "layer":
                    new_block = nn.ModuleList()
                    for layer_num, layer in enumerate(self.non_sequentialized_blocks[block]):
                        if (not dropout_all_blocks and block == len(self.non_sequentialized_blocks)-1 
                        and layer_num == len(self.non_sequentialized_blocks[block])-1):
                            new_block.append(layer)
                        else:
                            new_block.append(layer)
                            out_channels = self.non_sequentialized_blocks[block][layer_num].out_channels
                            dropout_layer = MCDropout(self.dropout_p) if (self.mask_type=="mc") else Masksembles1D(out_channels, self.num_masks, self.mask_scale)
                            new_block.append(dropout_layer)
                    self.non_sequentialized_blocks[block] = new_block
            # Overwrite blocks from super class
            for block in range(len(self.non_sequentialized_blocks)):
                self.blocks.append(nn.Sequential(*self.non_sequentialized_blocks[block]))

    def forward(self, x):
        output = x
        for block in self.blocks:
            output = block(output)
        if self.image_size == 224:
            output = self.avg_pool(output)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        self.intermediary_output_list = (output, [], 0, [])
        return [output]


# Heavily inspired from the ResNet version: https://github.com/hjdw2/Exit-Ensemble-Distillation/blob/main/resnet.py
class VGG19EarlyExit(VGG19):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args,  **kwargs)
        ex1conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        ex1conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        ex1conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        ex1bn1 = nn.BatchNorm2d(128)
        ex1bn2 = nn.BatchNorm2d(256)
        ex1bn3 = nn.BatchNorm2d(512)
        ex1relu1 = nn.ReLU(inplace=True)
        ex1relu2 = nn.ReLU(inplace=True)
        ex1relu3 = nn.ReLU(inplace=True)
        ex1featureextractor_list = nn.ModuleList(modules = [ex1conv1,ex1bn1,ex1relu1,ex1conv2,ex1bn2,ex1relu2,ex1conv3,ex1bn3,ex1relu3])
        self.ex1featureextractor = nn.Sequential(*ex1featureextractor_list)
        self.ex1linear = make_classifier(self.image_size, self.out_dim,mask_type=None) 

        ex2conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        ex2conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        ex2bn1 = nn.BatchNorm2d(256)
        ex2bn2 = nn.BatchNorm2d(512)
        ex2relu1 = nn.ReLU(inplace=True)
        ex2relu2 = nn.ReLU(inplace=True)
        ex2featureextractor_list = nn.ModuleList(modules = [ex2conv1,ex2bn1,ex2relu1,ex2conv2,ex2bn2,ex2relu2])
        self.ex2featureextractor = nn.Sequential(*ex2featureextractor_list)
        self.ex2linear = make_classifier(self.image_size, self.out_dim, mask_type=None) 


        ex3conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        ex3bn1 = nn.BatchNorm2d(512)
        ex3relu1 = nn.ReLU(inplace=True)
        ex3featureextractor_list = nn.ModuleList(modules = [ex3conv1,ex3bn1,ex3relu1])
        self.ex3featureextractor = nn.Sequential(*ex3featureextractor_list)
        self.ex3linear = make_classifier(self.image_size, self.out_dim, mask_type=None) 

        self.ex4linear = make_classifier(self.image_size, self.out_dim, mask_type=None) 

        self.init_weights()

    def forward(self, x):
        out = self.blocks[0](x)
        out1 = self.ex1featureextractor(F.relu(out))
        out1 = F.avg_pool2d(out1, 2)
        middle1_fea = out1
        out1 = out1.view(out1.size(0), -1)
        out1 = self.ex1linear(out1)
        out = self.blocks[1](out)
        out2 = self.ex2featureextractor(F.relu(out))
        out2 = F.avg_pool2d(out2, 2)
        middle2_fea = out2
        out2 = out2.view(out2.size(0), -1)
        out2 = self.ex2linear(out2)
        out = self.blocks[2](out)
        out3 = self.ex3featureextractor(F.relu(out))
        out3 = F.avg_pool2d(out3, 2)
        middle3_fea = out3
        out3 = out3.view(out3.size(0), -1)
        out3 = self.ex3linear(out3)
        out = self.blocks[3](out)
        out4 = F.avg_pool2d(out, 2)
        middle4_fea = out4
        out4 = out4.view(out4.size(0), -1)
        out4 = self.ex4linear(out4)
        out = self.blocks[4](out)
        # ExitEnsemble used avg_pool2d here, but the og didn't...
        final_fea = out
        out = out.view(out.size()[0], -1)
        out = self.classifier(out)
        self.intermediary_output_list = (out, [out1, out2, out3, out4], final_fea, [middle1_fea, middle2_fea, middle3_fea, middle4_fea])
        return [out1, out2, out3, out4, out]


class VGG19MCEarlyExit(VGG19EarlyExit):
    def __init__(self,dropout_exit = False, dropout = None, dropout_p = 0.5, n_exits = 4, 
                    out_dim = 100, mask_type = "mc", num_masks = 4, mask_scale = 4.0, *args,  **kwargs):
        super().__init__(n_exits = n_exits, out_dim = out_dim, *args,  **kwargs)
        self.n_exits = n_exits
        self.out_dim = out_dim
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.dropout_exit = dropout_exit
        self.mask_type = mask_type
        self.num_masks = num_masks
        self.mask_scale = mask_scale
        # Same as in MC
        dropout_all_blocks = False

        # Update exits so can get init correctly
        if self.dropout_exit:
            # Remake all exits with dropout
            self.ex1linear = make_classifier(self.image_size,self.out_dim,self.dropout_p,self.mask_type,self.num_masks,self.mask_scale)
            self.ex2linear = make_classifier(self.image_size,self.out_dim,self.dropout_p,self.mask_type,self.num_masks,self.mask_scale)
            self.ex3linear = make_classifier(self.image_size,self.out_dim,self.dropout_p,self.mask_type,self.num_masks,self.mask_scale)
            self.ex4linear = make_classifier(self.image_size,self.out_dim,self.dropout_p,self.mask_type,self.num_masks,self.mask_scale)
            # Overwrite classifier from super class
            self.classifier = make_classifier(self.image_size,self.out_dim,self.dropout_p,self.mask_type,self.num_masks,self.mask_scale)
        
        # Load weights
        self.init_weights()
        if self.image_size == 224 and self.out_dim == 7:
            load_imagenet_weights(self)
            dropout_all_blocks = True

        if self.dropout is not None:
            self.blocks = nn.ModuleList()
            for block in range(len(self.non_sequentialized_blocks)):
                if self.dropout == "block":
                    if not dropout_all_blocks and (block == len(self.non_sequentialized_blocks)-1 or block == len(self.non_sequentialized_blocks)-2):
                        pass
                    else:
                        out_channels = self.non_sequentialized_blocks[block][-2].out_channels
                        dropout_layer = MCDropout(self.dropout_p) if (self.mask_type=="mc") else Masksembles1D(out_channels, self.num_masks, self.mask_scale) 
                        self.non_sequentialized_blocks[block].append(dropout_layer)
                elif self.dropout == "layer":
                    new_block = nn.ModuleList()
                    for layer_num, layer in enumerate(self.non_sequentialized_blocks[block]):
                        if not dropout_all_blocks and (block == len(self.non_sequentialized_blocks)-1 
                            or block == len(self.non_sequentialized_blocks)-2) and layer_num == len(self.non_sequentialized_blocks[block])-1:
                            new_block.append(layer)
                        else:
                            new_block.append(layer)
                            out_channels = self.non_sequentialized_blocks[block][layer_num].out_channels
                            dropout_layer = MCDropout(self.dropout_p) if (self.mask_type=="mc") else Masksembles1D(out_channels, self.num_masks, self.mask_scale)
                            new_block.append(dropout_layer)
                    self.non_sequentialized_blocks[block] = new_block
            # Overwrite blocks from super class
            for block in range(len(self.non_sequentialized_blocks)):
                self.blocks.append(nn.Sequential(*self.non_sequentialized_blocks[block]))

class MCDropout(nn.Dropout):

    def forward(self, x):
        return F.dropout(x, self.p, True, self.inplace)


