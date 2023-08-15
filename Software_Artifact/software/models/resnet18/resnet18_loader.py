from utils import dict_drop
from models.resnet18.resnet18 import ResNet18EarlyExit, ResNet18Base, ResNet18MC, ResNet18MCEarlyExit

def get_res_net_18(ensemble, network_hyperparams):
    if ensemble == "early_exit" or ensemble is None:
        # n_exits, out_dim
        return ResNet18EarlyExit(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type","dropout","dropout_exit", "dropout_p", "mask_type", "num_masks", "mask_scale"))
    elif ensemble == "mc":
        # n_exits, out_dim, dropout, dropout_exit, dropout_p
        return ResNet18MC(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type"))
    elif ensemble == "mc_early_exit":
        # n_exits, out_dim, dropout, dropout_exit, dropout_p
        return ResNet18MCEarlyExit(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type"))