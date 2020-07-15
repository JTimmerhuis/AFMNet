# -*- coding: utf-8 -*-
"""
@author: Jardi
"""

from __future__ import print_function, division

from Dataclass import Dataclass
import nets

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from torch.optim import lr_scheduler
from torchvision import transforms, models
import time

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, scheduler = None, device = "cpu", num_epochs=25):
    """
    A method that trains the paramters of the network to classify AFM images.
    
    :param model: The network object
    :type model: Module
    :param criterion: The loss function used during training
    :type criterion: loss
    :param optimizer: The optimizer used during training
    :type optimizer: Optimizer
    :param dataloaders: Dictionary containing the Dataloader objects for the training and validation set
    :type dataloaders: dict
    :param dataset_sizes: Dictionary containing the size of both the training and validation set
    :type dataset_sizes: dict
    :param scheduler: The scheduler used during training, defaults to None
    :type scheduler: lr_scheduler, optional
    :param device: The device which is used to train the model, defaults to "cpu"
    :type device: device or str, optional
    :param num_epochs: Number of training epochs, defaults to 25
    :type num_epochs: int, optional
    :return: The trained network object
    :rtype: Module

    """
    since = time.time()

    ## Set first model as best model and initialize accuracy as 0% and loss as very high
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 100000

    ## Iterate over epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        ## Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            ## Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                ## Zero the parameter gradients
                optimizer.zero_grad()

                ## Forward pass
                ## Track history only if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    ## Perform a forward pass
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    ## Compute the loss
                    loss = criterion(outputs, labels)

                    ## Backward pass only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                ## Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            ## Perform a scheduling step in training phase if available (ReduceLROnPlateau scheduler excluded)
            if phase == 'train' and not isinstance(scheduler, (type(None), optim.lr_scheduler.ReduceLROnPlateau)):
                scheduler.step()
            
            ## Compute loss and accuracy of the epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            ## Deep copy the model if it has better validation accuracy than best model
            if phase == 'val' and epoch_acc > best_acc:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            ## Perform a scheduling step in the validation phase for the ReduceLROnPlateau scheduler
            if phase == 'val' and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(best_loss)

        print()

    ## Determine elapsed training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, device, dataloaders, class_names, imshow, num_images=9):
    """
    A method that shows a grid of images with their correct label and predictions. It also returns the validation accuracy.
    
    :param model: The network object
    :type model: Module
    :param device: The device on which the model is loaded
    :type device: device or str
    :param dataloaders: Dictionary containing the Dataloader objects for the training and validation set
    :type dataloaders: dict
    :param class_names: List of class names, i.e. 'good' and 'bad'
    :type class_names: list
    :param imshow: The imshow method of the Dataclass class
    :type imshow: method
    :param num_images: Number of images plotted in the grid, defaults to 9
    :type num_images: int, optional
    :return: The validation accuracy
    :rtype: float

    """
    
    ## Set model to evaluation mode
    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure()
    
    correct = 0
    total = 0

    ## Iterate over validation data
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            ## Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            ## Get the outputs and predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            ## Add the image to the grid
            for j in range(inputs.size()[0]):
                if images_so_far < num_images:
                    images_so_far += 1
                    ax = plt.subplot(num_images//3, 3, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}\nground truth: {}'.format(class_names[preds[j]],class_names[labels[j]]))
                    imshow(inputs.cpu().data[j])
                
                ## Check if the prediction was correct
                correct += (preds[j] == labels[j]).item()
                total += 1
        
        ## Set model to old mode (i.e. training or validations mode)
        model.train(mode=was_training)
        
        ## Determine validation accuracy
        val_acc = correct/total
        print('Val Acc: {:4f}'.format(val_acc))
        return val_acc