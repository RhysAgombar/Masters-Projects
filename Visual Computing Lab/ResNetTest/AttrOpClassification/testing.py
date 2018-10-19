# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 17:46:44 2018

@author: Raven
"""
import numpy as np
import torch.utils.data as tdata
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as tmodels
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torch.nn as nn
import scipy.io
from sklearn.model_selection import train_test_split
import random

data_dir = os.getcwd() + "\Dataset"
operators = ["Seasoned", "Sliced", "Whole"]
classes = ["Apple", "Potato", "Meat", "Carrot"]

data_transforms = {}
for i in operators:
    data_transforms[i] = transforms.Compose([ # Dataset for Training
        transforms.Resize(224),
        #transforms.RandomResizedCrop(224), # Random Resized Crop is not well suited for this dataset
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


feat_extractor = models.resnet18(pretrained=True)
feat_extractor.fc = nn.Sequential()

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in operators}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10,
                                             shuffle=True, num_workers=0)
              for x in operators}
dataset_sizes = {x: len(image_datasets[x]) for x in operators}   
    


def get_neg_pairs(pp, operators, objects):
    ls = []
    for j in operators:
        for k in objects:
            if j != pp[0] and k != pp[1]:
                ls.append([j,k])
    return ls

def get_batch(batch_size):
    batch = []
    while (len(batch) < batch_size):
        for op in operators: # for each operator...
            data = []
            
            img, inclasses = next(iter(dataloaders[op]))  # Gather a set of images and classes from them
    
            for i in range(len(img)):
                    
                np = get_neg_pairs([op, classes[inclasses[i]]], operators, classes)
                    
                ## Image, Object, Operator, nObject, nOperator
                data.append([Variable(feat_extractor(img[i].unsqueeze_(0))), int(inclasses[i]), int(operators.index(op)), np])
                    
                if(len(data) == batch_size):
                    break
                
            batch = batch + data
        
    return [ batch[i] for i in (random.sample(range(len(batch)), batch_size)) ]




print(get_batch(10))
















 
    
    
    