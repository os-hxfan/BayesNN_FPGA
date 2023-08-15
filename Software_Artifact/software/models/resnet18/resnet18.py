import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from utils import Masksembles2D, Masksembles1D

# Mostly adapted from https://github.com/hjdw2/Exit-Ensemble-Distillation/blob/main/resnet.py
# Main ResNet class, BasicBlock and conv3x3 are from above, while the remaining
# ResNet child classes are my own modifications of the ResNet class
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = F.relu(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=100):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.ex1conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.ex1conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.ex1conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.ex1bn1 = nn.BatchNorm2d(128)
        self.ex1bn2 = nn.BatchNorm2d(256)
        self.ex1bn3 = nn.BatchNorm2d(512)
        self.ex1linear = nn.Linear(512, num_classes)

        self.ex2conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.ex2conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.ex2bn1 = nn.BatchNorm2d(256)
        self.ex2bn2 = nn.BatchNorm2d(512)
        self.ex2linear = nn.Linear(512, num_classes)

        self.ex3conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.ex3bn1 = nn.BatchNorm2d(512)
        self.ex3linear = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)

        out1 = self.ex1bn1(self.ex1conv1(F.relu(out)))
        out1 = self.ex1bn2(self.ex1conv2(F.relu(out1)))
        out1 = self.ex1bn3(self.ex1conv3(F.relu(out1)))
        out1 = F.avg_pool2d(F.relu(out1), 4)
        middle1_fea = out1
        out1 = out1.view(out1.size(0), -1)
        out1 = self.ex1linear(out1)

        out = self.layer2(out)

        out2 = self.ex2bn1(self.ex2conv1(F.relu(out)))
        out2 = self.ex2bn2(self.ex2conv2(F.relu(out2)))
        out2 = F.avg_pool2d(F.relu(out2), 4)
        middle2_fea = out2
        out2 = out2.view(out2.size(0), -1)
        out2 = self.ex2linear(out2)

        out = self.layer3(out)

        out3 = self.ex3bn1(self.ex3conv1(F.relu(out)))
        out3 = F.avg_pool2d(F.relu(out3), 4)
        middle3_fea = out3
        out3 = out3.view(out3.size(0), -1)
        out3 = self.ex3linear(out3)

        out = self.layer4(out)

        out = F.avg_pool2d(F.relu(out), 4)
        final_fea = out
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        self.intermediary_output_list = (out, [out1, out2, out3], final_fea, [middle1_fea, middle2_fea, middle3_fea])
        return [out1, out2, out3, out]

class ResNet18EarlyExit(ResNet):
    def __init__(self,n_exits = 4, out_dim = 100, image_size = 32, *args,  **kwargs):
        super().__init__(block=BasicBlock, num_blocks=[2,2,2,2], num_classes=out_dim, *args,  **kwargs)
        self.n_exits = n_exits
        self.out_dim = out_dim


class ResNet18Base(ResNet):
    def __init__(self,n_exits = 1, out_dim = 100, *args,  **kwargs):
        super().__init__(block=BasicBlock, num_blocks=[2,2,2,2], num_classes=out_dim, *args,  **kwargs)
        self.n_exits = n_exits
        self.out_dim = out_dim
    
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(F.relu(out), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return [out]


class MCDropout(nn.Dropout):

    def forward(self, x):
        return F.dropout(x, self.p, True, self.inplace)

class ResNet18MC(ResNet):
    def __init__(self,dropout_exit = False, dropout = None, dropout_p = 0.5, n_exits = 1, out_dim = 100, image_size = 32, 
                    mask_type = "mc", num_masks = 4, mask_scale = 4.0, *args,  **kwargs):
        super().__init__(block=BasicBlock, num_blocks=[2,2,2,2], num_classes=out_dim, *args,  **kwargs)
        self.n_exits = n_exits
        self.out_dim = out_dim
        self.dropout_exit = dropout_exit
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.mask_type = mask_type
        self.num_masks = num_masks
        self.mask_scale = mask_scale
        layer_list = [self.layer1, self.layer2, self.layer3, self.layer4]
        if dropout == "block":
            for i in range(len(layer_list)):
                if i == len(layer_list)-1:
                    pass
                else:
                    if (self.mask_type == "mc"): layer_list[i] = nn.Sequential(layer_list[i], MCDropout(self.dropout_p))
                    else: layer_list[i] = nn.Sequential(layer_list[i], Masksembles2D(layer_list[i][-1].planes, self.num_masks, self.mask_scale))
            self.layer1, self.layer2, self.layer3, self.layer4 = layer_list
        elif dropout == "layer":
            for block in range(len(layer_list)):
                for layer in range(len(layer_list[block])):
                    if block == len(layer_list)-1 and layer == len(layer_list[block])-1:
                        pass
                    else:
                        if (self.mask_type == "mc"): layer_list[block][layer] = nn.Sequential(layer_list[block][layer], MCDropout(self.dropout_p))
                        else: layer_list[i] = nn.Sequential(layer_list[i], Masksembles2D(layer_list[block][layer].planes, self.num_masks, self.mask_scale))
        if self.dropout_exit:
            if (self.mask_type == "mc"): self.exit_dropout = MCDropout(self.dropout_p)
            else: self.exit_dropout =  Masksembles1D(512 * BasicBlock.expansion, self.num_masks, self.mask_scale)

    
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(F.relu(out), 4)
        out = out.view(out.size(0), -1)
        if self.dropout_exit:
            out = self.exit_dropout(out)
        out = self.linear(out)
        self.intermediary_output_list = (out, [], 0, [])
        return [out]

class ResNet18MCEarlyExit(ResNet):
    def __init__(self,dropout_exit = False, dropout = None, dropout_p = 0.5, n_exits = 4, out_dim = 100, image_size = 32, 
                    mask_type = "mc", num_masks = 4, mask_scale = 4.0, *args,  **kwargs):
        super().__init__(block=BasicBlock, num_blocks=[2,2,2,2], num_classes=out_dim, *args,  **kwargs)
        self.n_exits = n_exits
        self.out_dim = out_dim
        self.dropout_exit = dropout_exit
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.mask_type = mask_type
        self.num_masks = num_masks
        self.mask_scale = mask_scale
        layer_list = [self.layer1, self.layer2, self.layer3, self.layer4]
        if dropout == "block":
            for i in range(len(layer_list)):
                if i == len(layer_list)-1:
                    pass
                else:
                    if (self.mask_type == "mc"): layer_list[i] = nn.Sequential(layer_list[i], MCDropout(self.dropout_p))
                    else: layer_list[i] = nn.Sequential(layer_list[i], Masksembles2D(layer_list[i][-1].planes, self.num_masks, self.mask_scale))
            self.layer1, self.layer2, self.layer3, self.layer4 = layer_list
        elif dropout == "layer":
            for block in range(len(layer_list)):
                for layer in range(len(layer_list[block])):
                    if block == len(layer_list)-1 and layer == len(layer_list[block])-1:
                        pass
                    else:
                        if (self.mask_type == "mc"): layer_list[block][layer] = nn.Sequential(layer_list[block][layer], MCDropout(self.dropout_p))
                        else: layer_list[i] = nn.Sequential(layer_list[i], Masksembles2D(layer_list[block][layer].planes, self.num_masks, self.mask_scale))

        if self.dropout_exit:
            if (self.mask_type == "mc"): 
                self.exit1_dropout = MCDropout(self.dropout_p)
                self.exit2_dropout = MCDropout(self.dropout_p)
                self.exit3_dropout = MCDropout(self.dropout_p)
                self.exit_dropout = MCDropout(self.dropout_p)
            else: 
                self.exit1_dropout =  Masksembles1D(512, self.num_masks, self.mask_scale)
                self.exit2_dropout =  Masksembles1D(512, self.num_masks, self.mask_scale)
                self.exit3_dropout =  Masksembles1D(512, self.num_masks, self.mask_scale)
                self.exit_dropout =  Masksembles1D(512 * BasicBlock.expansion, self.num_masks, self.mask_scale)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)

        out1 = self.ex1bn1(self.ex1conv1(F.relu(out)))
        out1 = self.ex1bn2(self.ex1conv2(F.relu(out1)))
        out1 = self.ex1bn3(self.ex1conv3(F.relu(out1)))
        out1 = F.avg_pool2d(F.relu(out1), 4)
        middle1_fea = out1
        out1 = out1.view(out1.size(0), -1)
        if self.dropout_exit:
            out1 = self.exit1_dropout(out1)
        out1 = self.ex1linear(out1)

        out = self.layer2(out)

        out2 = self.ex2bn1(self.ex2conv1(F.relu(out)))
        out2 = self.ex2bn2(self.ex2conv2(F.relu(out2)))
        out2 = F.avg_pool2d(F.relu(out2), 4)
        middle2_fea = out2
        out2 = out2.view(out2.size(0), -1)
        if self.dropout_exit:
            out2 = self.exit2_dropout(out2)
        out2 = self.ex2linear(out2)

        out = self.layer3(out)

        out3 = self.ex3bn1(self.ex3conv1(F.relu(out)))
        out3 = F.avg_pool2d(F.relu(out3), 4)
        middle3_fea = out3
        out3 = out3.view(out3.size(0), -1)
        if self.dropout_exit:
            out3 = self.exit3_dropout(out3)
        out3 = self.ex3linear(out3)

        out = self.layer4(out)

        out = F.avg_pool2d(F.relu(out), 4)
        final_fea = out
        out = out.view(out.size(0), -1)
        if self.dropout_exit:
            out = self.exit_dropout(out)
        out = self.linear(out)
        self.intermediary_output_list = (out, [out1, out2, out3], final_fea, [middle1_fea, middle2_fea, middle3_fea])
        return [out1, out2, out3, out]