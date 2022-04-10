import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_training_dataloader(train_path,mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], batch_size=256, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    training_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
    training_loader = DataLoader(
        training_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return training_loader

def get_test_dataloader(test_path,mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], batch_size=256, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    test_dataset = torchvision.datasets.ImageFolder(root=test_path,transform=transform_test)
    test_loader = DataLoader(
        test_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader



