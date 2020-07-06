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
    data_dir = 'data'
    batch_size = 20
    num_epochs = 50
    num_workers = 4
    load_val_acc = 90
    arch = models.resnet18
    
    net = nets.FineNet(arch)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #scheduler = None
    
    ## ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    ## Own Dataset
#    mean = [0.741, 0.429, 0.183]
#    std = [0.113, 0.157, 0.079]
    
    transforms_train = [Dataclass.AFMcrop(), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    transforms_val = [Dataclass.AFMcrop(), transforms.Resize(256), transforms.CenterCrop(224)]
    
    
    data = Dataclass(data_dir)
    data.add_transforms(transforms_train, transforms_val, mean, std)
    data.load_data(batch_size, num_workers)
    
    #net = train_model(net, criterion, optimizer, data.dataloaders, data.dataset_sizes, scheduler, device, num_epochs)
    nets.load(net, None, optimizer, mean, std, num_epochs, load_val_acc, scheduler, batch_size, device)
    val_acc = visualize_model(net, device, data.dataloaders, data.class_names, data.imshow)
    
    #nets.save(net, optimizer, mean, std, num_epochs, val_acc, scheduler, batch_size = batch_size)