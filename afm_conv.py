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
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 100000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and not isinstance(scheduler, (type(None), optim.lr_scheduler.ReduceLROnPlateau)):
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == 'val' and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(best_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, dataloaders, class_names, imshow, num_images=9):
    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure()
    
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if images_so_far < num_images:
                    images_so_far += 1
                    ax = plt.subplot(num_images//3, 3, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}\nground truth: {}'.format(class_names[preds[j]],class_names[labels[j]]))
                    imshow(inputs.cpu().data[j])
                
                correct += (preds[j] == labels[j]).item()
                total += 1
        model.train(mode=was_training)
        val_acc = correct/total
        print('Val Acc: {:4f}'.format(val_acc))
        return val_acc

if __name__ == "__main__":
    data_dir = 'data'
    batch_size = 20
    num_epochs = 50
    num_workers = 4
    load_val_acc = 62
    arch = models.resnet18
    
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #inp, classes = data.get_batchdata()
    #data.imshow(inp)#, title=[data_loader.class_names[x] for x in classes])
    
    net = nets.FineNet(arch)
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scheduler = None
    
    net = train_model(net, criterion, optimizer, data.dataloaders, data.dataset_sizes, scheduler, device, num_epochs)
    #nets.load(net, None, optimizer, mean, std, num_epochs, load_val_acc, scheduler, device)
    val_acc = visualize_model(net, data.dataloaders, data.class_names, data.imshow)
    
    nets.save(net, optimizer, mean, std, num_epochs, val_acc, scheduler, batch_size = batch_size)