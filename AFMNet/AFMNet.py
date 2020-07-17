# -*- coding: utf-8 -*-
"""
@author: Jardi
"""

from __future__ import print_function, division

import sys
sys.path.insert(1, 'code')

from Dataclass import Dataclass
import nets
from functions import train_model, visualize_model

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

if __name__ == "__main__":
    ## Set data directory
    data_dir = 'data'
    
    ## Initialize batch size, number of epochs
    batch_size = 20
    num_epochs = 50
    
    ## Initialize number of parallel workers for parallel processing
    num_workers = 4
    
    ## If you want to load a model, set the accuracy of the model you want to load
    load_val_acc = 90
    
    ## Set the architecture of the fine-tuned net (FineNet) or fixed feature extractor (FeatureNet)
    ## Currently, only the resnet18 architecture is supported
    arch = models.resnet18
    
    ## Initialize net
    #net = nets.ConvNet()
    net = nets.FineNet(arch)
    #net = nets.FeatureNet(arch)
    
    ## If a CUDA GPU is available set the device to GPU, otherwise the CPU is used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    ## Set the loss function (criterion), optimizer and scheduler
    ## Cross Entropy Loss is used as a default
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #scheduler = None
    
    ## Initialize the mean and standard deviation used to normalize the images
    ## Use either the ImageNet mean + std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    ## Or use the mean + std from our own dataset
    #mean = [0.741, 0.429, 0.183]
    #std = [0.113, 0.157, 0.079]
    
    ## Initialize the transforms used to preprocess the training and validation data set
    transforms_train = [Dataclass.AFMcrop(), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    transforms_val = [Dataclass.AFMcrop(), transforms.Resize(256), transforms.CenterCrop(224)]
    
    ## Initialize the data set and add the final transforms to preprocess the data
    data = Dataclass(data_dir)
    data.add_transforms(transforms_train, transforms_val, mean, std)
    
    ## Load and preprocess the data
    data.load_data(batch_size, num_workers)
    
    ## Either train or load the model
    #net = train_model(net, criterion, optimizer, data.dataloaders, data.dataset_sizes, scheduler, device, num_epochs)
    nets.load(net, None, optimizer, mean, std, num_epochs, load_val_acc, scheduler, batch_size, device)
    
    ## Check the performance of the model and visualize the prediction of some images
    val_acc = visualize_model(net, device, data.dataloaders, data.class_names, data.imshow)
    
    ## Save the model if you want (be aware to not override any existing model!)
    #nets.save(net, optimizer, mean, std, num_epochs, val_acc, scheduler, batch_size = batch_size)