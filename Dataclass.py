# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:57:22 2019

@author: jardi
"""

import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
import copy
import math
from torchvision import datasets, transforms

class Dataclass():
    class AFMcrop(object):
        def __call__(self, img): 
            w, h = img.size
            new_w = math.floor(0.7*w)

            return transforms.functional.crop(img, 0, 0, h, new_w)
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def add_transforms(self, transforms_train, transforms_val, mean, std):
        self.mean = mean
        self.std = std
        transforms_train.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        transforms_val.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.data_transforms = {
            'train': transforms.Compose(transforms_train),
            'val': transforms.Compose(transforms_val),
            #'test': transforms.Compose(transforms_val),
        }
        
    def load_data(self, batch_size = 1, num_workers = 0):
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x]) for x in ['train', 'val']}
        try:
            self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
            self.get_batchdata()
        except RuntimeError:
            print("Using multiple workers failed. Using 0 workers instead. Using Spyder with Windows might cause this problem.")
            self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes
        
    def get_meanstd(self, transform, batch_size, dataset = 'train'):
        mean = 0.
        std = 0.
        nb_samples = 0.
        tform = copy.deepcopy(transform)
        tform.extend([transforms.ToTensor()])
        tform = transforms.Compose(tform)
        dataset = datasets.ImageFolder(os.path.join(self.data_dir, dataset), tform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for data in loader:
            batch_samples = data[0].size(0)
            data = data[0].view(batch_samples, data[0].size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples
        
        mean /= nb_samples
        std /= nb_samples
        return mean, std
        
    def get_batchdata(self, data = 'train'):
        inputs, classes = next(iter(self.dataloaders[data]))
        return torchvision.utils.make_grid(inputs), classes
        
    def imshow(self, inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        inp = self.std * inp + self.mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated