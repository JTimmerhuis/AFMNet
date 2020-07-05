# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 14:08:04 2019

@author: jardi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import models

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FineNet(models.ResNet):
    def __init__(self, arch):
        if arch == models.resnet18:
            block = models.resnet.BasicBlock
            layers = [2, 2, 2, 2]
        super(FineNet, self).__init__(block, layers)
        model = arch(pretrained=True)
        self.arch = arch.__name__
        self.load_state_dict(model.state_dict())
        self._edit_param()
        
    def _edit_param(self):
        num_ftrs = self.fc.in_features
        self.fc = nn.Linear(num_ftrs, 2)
        
class FeatureNet(FineNet):
    def __init__(self, arch):
        super(FeatureNet, self).__init__(arch)
        
    def _edit_param(self):
        self._freeze_net()
        super(FeatureNet, self)._edit_param()
        
    def _freeze_net(self):
        for param in self.parameters():
            param.requires_grad = False
       
def get_path(net, optimizer, mean, std, epochs, val_acc, scheduler = None, batch_size = 1):
    net_string = type(net).__name__
    if net_string.__eq__("FineNet") or net_string.__eq__("FeatureNet"):
        arch_string = net.arch + "_"
    else:
        arch_string = ""
    opt_string = type(optimizer).__name__
    if batch_size > 1:
        if opt_string.__eq__("SGD"):
            opt_string = "BGD" + str(batch_size)
        else:
            opt_string = "Batch" + opt_string + str(batch_size)
            
    if scheduler is None:
        sched_string = "NoScheduler"
    else:
        sched_string = type(scheduler).__name__
    
    if mean == [0.485, 0.456, 0.406] and std == [0.229, 0.224, 0.225]:
        mstd_string = "ImageMStd"
    elif mean == [0.741, 0.429, 0.183] and std == [0.113, 0.157, 0.079]:
        mstd_string = "OwnMStd"
    else:
        mstd_string = "OtherMStd"
    
    val_acc = round(val_acc*100)
    return "models" + os.sep + net_string + os.sep + arch_string + opt_string + "_" + sched_string + "_" + mstd_string + "_Eps" + str(epochs) + "_Val" + str(val_acc) + '.pt'

def save(net, optimizer, mean, std, epochs, val_acc, scheduler = None, batch_size = 1):    
    path = get_path(net, optimizer, mean, std, epochs, val_acc, scheduler, batch_size)
    save_dict = {
            'state_dict': net.state_dict(),
            'mean': mean,
            'std': std,
            }
    torch.save(save_dict, path)
    
def load(net, path = None, optimizer = None, mean = None, std = None, epochs = None, val_acc = None, scheduler = None, batch_size = 1, device = "cpu"):
    if path is None:
        path = get_path(net, optimizer, mean, std, epochs, val_acc/100, scheduler, batch_size)
    load_dict = torch.load(path, map_location=device)
    net.load_state_dict(load_dict['state_dict'])
    mean = load_dict['mean']
    std = load_dict['std']
    return mean, std