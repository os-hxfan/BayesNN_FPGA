import torch
import torchvision
import torchvision.transforms as transforms

def get_transforms(dataset_name = None, type = None):
    """Define the standard transforms for easy access"""
    if dataset_name == "cifar10":
        transform_list = [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    else:
        transform_list = [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    return transform

