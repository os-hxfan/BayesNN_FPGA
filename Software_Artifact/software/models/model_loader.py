import torch
from utils import dict_drop
from models.resnet18 import get_res_net_18
from models.vgg19 import get_vgg_19

# Inspired from how https://github.com/mary-phuong/multiexit-distillation/blob/master/main.py 
# load their networks
def get_network(network_hyperparams):
    if network_hyperparams["load_model"] is not None:
        if torch.cuda.is_available():
            try:
                device = network_hyperparams["gpu_device"]
                model = torch.load(network_hyperparams["load_model"], map_location = device)
            except KeyError:
                model = torch.load(network_hyperparams["load_model"])
        else:
            model = torch.load(network_hyperparams["load_model"], map_location=torch.device('cpu'))
    elif network_hyperparams["call"] == "ResNet18":
        model = get_res_net_18(network_hyperparams["resnet_type"], dict_drop(network_hyperparams, "call", "load_model","resnet_type"))
    elif network_hyperparams["call"] == "VGG19":
        model = get_vgg_19(network_hyperparams["resnet_type"], dict_drop(network_hyperparams, "call", "load_model","resnet_type"))    
    else:
        raise AttributeError
    return model