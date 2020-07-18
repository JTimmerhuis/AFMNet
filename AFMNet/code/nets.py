# -*- coding: utf-8 -*-
"""
Contains all network classes and methods relevant to loading and saving network parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import models

class ConvNet(nn.Module):
    """Class for our convolutional neural network with two convolutional layers as designed in the report."""
    
    def __init__(self):
        """
        Initializes a ConvNet with two convolutional layers and three fully connected layers.
        

        Returns:
            None.

        """

        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        """A forward pass of the network. Called when using net(input), so use net(input), not net.forward(input)!!

        Args:
          x(Tensor): Input tensor.

        Returns:
          Tensor: Output Tensor.

        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FineNet(models.ResNet):
    """Class for the Finetuned (resnet18) model."""
    
    def __init__(self, arch = models.resnet18):
        """
        Initializes a fine tuned ResNet.
        

        Args:
            arch (function, optional): Model architecture. Defaults to models.resnet18.

        Raises:
            NotImplementedError: Exception for architectures that are not implemented yet.

        Returns:
            None.

        """
        
        ## Initializing a random ResNet
        if arch == models.resnet18:
            block = models.resnet.BasicBlock
            layers = [2, 2, 2, 2]
        else:
            raise NotImplementedError("This architecture is not implemented yet!")
        super(FineNet, self).__init__(block, layers)
        
        ## Loading pretrained parameters
        model = arch(pretrained=True)
        self.arch = arch.__name__
        self.load_state_dict(model.state_dict())
        
        ## Replacing last layer
        self.edit_param()
        
    def edit_param(self):
        """
        Replaces the last layer of the network to correspond with 2 classes.
        
        Returns:
            None.

        """

        ## Number of input features
        num_ftrs = self.fc.in_features
        
        ## Replacing last layer
        self.fc = nn.Linear(num_ftrs, 2)
        
class FeatureNet(FineNet):
    """Class for the model with fixed paramters used as a feature extractor."""
    
    def __init__(self, arch = models.resnet18):
        """
        Initializes a ResNet to be used as a fixed feature extractor.

        Args:
            arch (function, optional): Model architecture. Defaults to models.resnet18.

        Returns:
            None.

        """
        
        ## Runs the init of FineNet with the _edit_param function of this class
        super(FeatureNet, self).__init__(arch)
        
    def edit_param(self):
        """
        Freezes the network parameters and replaces the last layer.

        Returns:
            None.

        """
        
        ## Freezes the network parameters
        self.freeze_net()
        
        ## Replaces the last layer
        super(FeatureNet, self).edit_param()
        
    def freeze_net(self):
        """
        Freezes the network paramters.

        Returns:
            None.

        """

        for param in self.parameters():
            param.requires_grad = False
            
            
def save(net, optimizer, mean, std, epochs, val_acc, scheduler = None, batch_size = 1):    
    """Method to save the network parameters, mean and standard deviation.

    Args:
      net(Module): The network object.
      optimizer(Optimizer): The optimizer used during training.
      mean(list): Means of each channel used to normalize the data.
      std(list): Standard deviations of each channel used to normalize the data.
      epochs(int): Number of epochs used during training.
      val_acc(float): The validation accuracy of the model.
      scheduler(lr_scheduler, optional): The scheduler used during training. Defaults to None.
      batch_size(int, optional): The batch size used during training. Defaults to 1.

    Returns:
        None.

    """
    
    ## Get the path to save the file top
    path = get_path(net, optimizer, mean, std, epochs, val_acc, scheduler, batch_size)
    
    ## Create a dictionary with data to save and save data
    save_dict = {
            'state_dict': net.state_dict(),
            'mean': mean,
            'std': std,
            }
    torch.save(save_dict, path)
    
def load(net, path = None, optimizer = None, mean = None, std = None, epochs = None, val_acc = None, scheduler = None, batch_size = 1, device = "cpu"):
    """Method that loads a network from a previously saved file. You can either pass a path string or pass all the other variables to the method. Returns means and standard deviations and loads network parameters to the net object.

    Args:
      net(Module): The network object.
      path(str, optional): The path to the model file. Defaults to None.
      optimizer(Optimizer, optional): The optimizer used during training. Defaults to None.
      mean(list, optional): Means of each channel used to normalize the data. Defaults to None.
      std(list, optional): Standard deviations of each channel used to normalize the data. Defaults to None.
      epochs(int, optional): Number of epochs used during training. Defaults to None.
      val_acc(float, optional): The validation accuracy of the model. Defaults to None.
      scheduler(lr_scheduler, optional): The scheduler used during training. Defaults to None.
      batch_size(int, optional): The batch size used during training. Defaults to 1.
      device(device: device or str, optional): The device on which the network is loaded. Defaults to "cpu".

    Returns:
      tuple: Tuple of lists containt the means and standard deviations of each channel.

    """
    
    ## If no path is passed to the method, get the path
    if path is None:
        path = get_path(net, optimizer, mean, std, epochs, val_acc/100, scheduler, batch_size)
        
    ## Load the dictionary
    load_dict = torch.load(path, map_location=device)
    
    ## Load the network parameters
    net.load_state_dict(load_dict['state_dict'])
    
    ## Load the means and standard deviations
    mean = load_dict['mean']
    std = load_dict['std']
    return mean, std
       
def get_path(net, optimizer, mean, std, epochs, val_acc, scheduler = None, batch_size = 1):
    """Generates a unique path string for a model with the given paramters.

    Args:
      net(Module): The network object.
      optimizer(Optimizer): The optimizer used during training.
      mean(list): Means of each channel used to normalize the data.
      std(list): Standard deviations of each channel used to normalize the data.
      epochs(int): Number of epochs used during training.
      val_acc(float): The validation accuracy of the model.
      scheduler(lr_scheduler, optional): The scheduler used during training. Defaults to None.
      batch_size(int, optional): The batch size used during training. Defaults to 1.

    Returns:
      str: The path string of the model.

    """
    
    ## Convert the network (architecture) to a string
    net_string = type(net).__name__
    if net_string.__eq__("FineNet") or net_string.__eq__("FeatureNet"):
        arch_string = net.arch + "_"
    else:
        arch_string = ""
    
    ## Convert the optimizer to a string
    opt_string = type(optimizer).__name__
    if batch_size > 1:
        if opt_string.__eq__("SGD"):
            opt_string = "BGD" + str(batch_size)
        else:
            opt_string = "Batch" + opt_string + str(batch_size)
    
    ## Convert the scheduler to a string
    if scheduler is None:
        sched_string = "NoScheduler"
    else:
        sched_string = type(scheduler).__name__
    
    ## Convert the used means and standard deviations to a string
    if mean == [0.485, 0.456, 0.406] and std == [0.229, 0.224, 0.225]:
        mstd_string = "ImageMStd"
    elif mean == [0.741, 0.429, 0.183] and std == [0.113, 0.157, 0.079]:
        mstd_string = "OwnMStd"
    else:
        mstd_string = "OtherMStd"
    
    ## Convert the validation accuracy to an int
    val_acc = round(val_acc*100)
    
    ## Combine all strings in one path file
    return "models" + os.sep + net_string + os.sep + arch_string + opt_string + "_" + sched_string + "_" + mstd_string + "_Eps" + str(epochs) + "_Val" + str(val_acc) + '.pt'