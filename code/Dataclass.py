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
            """
            AFMCrop is a transform that removes the right 30% of the image. AFM images often have a colour scale bar at the right.
            
            :param img: An AFM image with a colour scale bar at the right
            :type img: PIL Image
            :return: PIL Image with the right 30% cropped
            :rtype: PIL Image

            """
            
            w, h = img.size
            new_w = math.floor(0.7*w)

            return transforms.functional.crop(img, 0, 0, h, new_w)
    
    def __init__(self, data_dir):
        """
        Initalized Dataclass and sets data directory.
        
        :param data_dir: Directory of the data folder
        :type data_dir: str

        """
        self.data_dir = data_dir
        
    def add_transforms(self, transforms_train, transforms_val, mean, std):
        """
        Adds transforms that converts image to tensor and normalizes image for preprocessing.
        
        :param transforms_train: List of image transforms performed on the training data set
        :type transforms_train: list
        :param transforms_val: List of image transforms performed on the validation data set
        :type transforms_val: list
        :param mean: Means of each channel used to normalize the data
        :type mean: list
        :param std: Standard deviations of each channel used to normalize the data
        :type std: list


        """
        self.mean = mean
        self.std = std
        
        ## Add the transforms that convert the data to tensor and normalized the images.
        transforms_train.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        transforms_val.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        
        ## Set the transforms in a dictionary.
        self.data_transforms = {
            'train': transforms.Compose(transforms_train),
            'val': transforms.Compose(transforms_val),
        }
        
    def load_data(self, batch_size = 1, num_workers = 0):
        """
        A method that loads the data in a DataLoade object.
        
        :param batch_size: The batch size used during training, defaults to 1
        :type batch_size: int, optional
        :param num_workers: Number of parallel workers, defaults to 0
        :type num_workers: int, optional

        """
        
        ## Get the data from the data directory
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x]) for x in ['train', 'val']}
        
        ## Load the data in a DataLoader object
        try:
            self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
            self.get_batchdata()
        except RuntimeError:
            print("Using multiple workers failed. Using 0 workers instead. Using Spyder with Windows might cause this problem.")
            self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}
        
        ## Store the sizes of the data set and class names ('Good' and 'Bad' in our case)
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes
        
    def get_meanstd(self, transform, batch_size, dataset = 'train'):
        """
        Calculates the mean and standard deviation of a data set.
        
        :param transform: List of image transforms used in preprocessing
        :type transform: list
        :param batch_size: Size of the batch in the dataloader
        :type batch_size: int
        :param dataset: String of dataset, either 'train' or 'val', defaults to 'train'
        :type dataset: str, optional
        :return: Tuple of mean and standard deviation
        :rtype: tuple

        """
        mean = 0.
        std = 0.
        nb_samples = 0.
        
        ## Copying the transforms and converting the image to a tensor
        tform = copy.deepcopy(transform)
        tform.extend([transforms.ToTensor()])
        tform = transforms.Compose(tform)
        
        ## Loading the data set
        dataset = datasets.ImageFolder(os.path.join(self.data_dir, dataset), tform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        ## Iterating over data in loader and summing the means and std's of each image
        for data in loader:
            batch_samples = data[0].size(0)
            data = data[0].view(batch_samples, data[0].size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples
        
        ## Dividing by number of images to get mean and std
        mean /= nb_samples
        std /= nb_samples
        return mean, std
        
    def get_batchdata(self, data = 'train'):
        """
        Old method that makes a grid of images.
        
        :param data: String of the data set, either 'train' or 'val', defaults to 'train'
        :type data: str, optional
        :return: Tuple of Image grid and classes
        :rtype: tuple

        """
        inputs, classes = next(iter(self.dataloaders[data]))
        return torchvision.utils.make_grid(inputs), classes
        
    def imshow(self, inp, title=None):
        """
        Plot a tensor as an image.
        
        :param inp: An image tensor.
        :type inp: Tensor
        :param title: Title of the plot, defaults to None
        :type title: str, optional

        """
        
        ## Convert the tensor back to an image
        inp = inp.numpy().transpose((1, 2, 0))
        inp = self.std * inp + self.mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated