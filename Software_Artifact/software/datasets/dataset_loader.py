import torch
import torchvision
import os
import random
import PIL
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_dataloader(hyperparameters, random_seed = None):
    if random_seed is None:
        random_seed = random.randint(1,1000)
    dataset_loader = DatasetLoader(dataset_name = hyperparameters["dataset_name"], 
        batch_size = hyperparameters["batch_size"], augment = hyperparameters["augment"], 
        random_seed = random_seed, valid_split = hyperparameters["val_split"])
    train_loader, val_loader, test_loader = dataset_loader.get_dataloaders()
    return train_loader, val_loader, test_loader 

# Adapted from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
# General structure is the same but is changed for the datasets and transformations I need
class DatasetLoader:
    def __init__(self,dataset_name= "cifar10", batch_size = (64,250,250), augment = True, random_seed = 42, valid_split = 0.2):
        self.dataset_name = dataset_name
        self.train_batch_size = batch_size[0]
        self.val_batch_size = batch_size[1]
        self.test_batch_size = batch_size[2]
        self.augment = augment
        self.valid_size = valid_split
        self.random_seed = random_seed
        self.data_dir = "./datasets/data/"+self.dataset_name
        self.size = 32
        if dataset_name == "imagenet" or dataset_name == "chestx":
            self.size = 224
        self._get_transforms()
        self._get_dataset()
        if self._sampler_needed():
            self._get_samplers()
        else:
            self.train_sampler = None

    def _sampler_needed(self):
        datasets_without_val = ["cifar10","cifar100","imagenet"]
        if self.dataset_name in datasets_without_val:
            self.shuffle = False
            return True
        else:
            self.shuffle = True
            return False

    def _get_transforms(self):
        if self.dataset_name == "cifar10":
            self.mean=[0.4914, 0.4822, 0.4465]
            self.std=[0.2023, 0.1994, 0.2010]
        elif self.dataset_name == "cifar100":
            self.mean=[0.5071, 0.4865, 0.4409]
            self.std=[0.2673, 0.2564, 0.2762]
        elif self.dataset_name == "chestx":
            # From here: https://github.com/arnoweng/CheXNet/blob/master/model.py
            # Using imagenet as using pretrained weights
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        normalize = transforms.Normalize(mean = self.mean,std = self.std)   
        if self.size == 224:
            self.val_transforms = transforms.Compose([
                transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                ])
            self.test_transforms = transforms.Compose([
                transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                ])
            if self.augment:
                self.train_transforms = transforms.Compose([
                transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
                ])                
            else:
                self.train_transforms = transforms.Compose([
                    transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
        else:
            self.val_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
            self.test_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
            if self.augment:
                self.train_transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    normalize,
                ])       
            else:
                self.train_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
        return None

    def _get_dataset(self):
        #print(os.getcwd())  #MultiExit_BNNS/
        if self.dataset_name == "cifar10":
            self.train_set = datasets.CIFAR10(
                root=self.data_dir, train=True,
                download=True, transform=self.train_transforms,
            )
            self.val_set = datasets.CIFAR10(
                root=self.data_dir, train=True,
                download=False, transform=self.val_transforms,
            )
            self.test_set = datasets.CIFAR10(
                root=self.data_dir, train=False,
                download=False, transform=self.test_transforms,
            )
        elif self.dataset_name == "cifar100":
            self.train_set = datasets.CIFAR100(
                root=self.data_dir, train=True,
                download=True, transform=self.train_transforms,
            )
            self.val_set = datasets.CIFAR100(
                root=self.data_dir, train=True,
                download=False, transform=self.val_transforms,
            )
            self.test_set = datasets.CIFAR100(
                root=self.data_dir, train=False,
                download=False, transform=self.test_transforms,
            )
        return None
    
    def _get_samplers(self):
        num_train = len(self.train_set)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)
        return None

    def get_dataloaders(self):
        if self.train_sampler is not None:
            train_loader = torch.utils.data.DataLoader(self.train_set,batch_size = self.train_batch_size, sampler=self.train_sampler,
                num_workers=1, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(self.val_set,batch_size = self.val_batch_size, sampler=self.valid_sampler,
                num_workers=1, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(self.train_set,batch_size = self.train_batch_size, shuffle = self.shuffle,
                num_workers=1, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(self.val_set,batch_size = self.val_batch_size, shuffle = self.shuffle,
                num_workers=1, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(self.test_set,batch_size = self.test_batch_size, pin_memory = True)
        return train_loader, val_loader, test_loader

