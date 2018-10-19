from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import Model, DataProcessor


data_dir = os.getcwd() + "\Dataset"

operators = []
objects = []
dirs = os.listdir(data_dir)
operators = dirs

for i in operators:
    for j in os.listdir(data_dir + "\\" + i):
        objects.append(j)

objects = list(set(objects))
operators = list(set(operators))

print(objects)
print(operators)

data_transforms = {}
for i in operators:
    data_transforms[i] = transforms.Compose([ # Dataset for Training
        transforms.Resize(224),
        #transforms.RandomResizedCrop(224), # Random Resized Crop is not well suited for this dataset
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

print(data_transforms)

##################################

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in operators}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10,
                                             shuffle=True, num_workers=0)
              for x in operators}
dataset_sizes = {x: len(image_datasets[x]) for x in operators}

##################################

image_testsets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in operators}

testloaders = {x: torch.utils.data.DataLoader(image_testsets[x], batch_size=10,
                                             shuffle=True, num_workers=0)
              for x in operators}
testset_sizes = {x: len(image_testsets[x]) for x in operators}

##################################

data = [objects, operators]
model = Model.AttrOpModel(data)

attr_params = [param for name, param in model.named_parameters() if 'attr_op' in name and param.requires_grad]
other_params = [param for name, param in model.named_parameters() if 'attr_op' not in name and param.requires_grad]
optim_params = [{'params':attr_params, 'lr':1e-05}, {'params':other_params}]

optimizer = optim.Adam(optim_params, lr=1e-04, weight_decay=5e-5)
feat_extractor = models.resnet18(pretrained=True)
feat_extractor.fc = nn.Sequential()


## Just Apple for now
inApple = []
while len(inApple) < 5:
    inputs, classes = next(iter(dataloaders['Whole']))
    for i in range(len(inputs)):
        if classes.data[i] == objects.index("Apple"):
            inApple.append(inputs[i])

af = [] # Apple Features
for i in range(len(inApple)):
    af.append(feat_extractor(inApple[i].unsqueeze_(0)))
    
    

def get_batch(batch_size):
    batch = []
    while (len(batch) < batch_size):
        for op in operators: # for each operator...
            data = []
            
            img, inclasses = next(iter(dataloaders[op]))  # Gather a set of images and classes from them
    
            for i in range(len(img)):
                    
                np = DataProcessor.get_neg_pairs([op, classes[inclasses[i]]], operators, classes)
                    
                ## Image, Object, Operator, nObject, nOperator
                data.append([Variable(feat_extractor(img[i].unsqueeze_(0))), int(inclasses[i]), int(operators.index(op)), np])
                    
                if(len(data) == batch_size):
                    break
                
            batch = batch + data
        
    return [ batch[i] for i in (random.sample(range(len(batch)), batch_size)) ]



for epoch in range(0,5):
    batch_size = 20
    loss = 0
    for op in operators: # for each operator...
        class_names = image_datasets[op].classes
        imgs = []
        pp = []
        np = []
        
        while len(imgs) < batch_size:
            inputs, classes = next(iter(dataloaders[op]))  # Gather a set of images and classes from them
            
            for i in inputs:
                imgs.append(Variable(feat_extractor(i.unsqueeze_(0))))
                if(len(imgs) == batch_size):
                    break
            for i in classes:
                pp.append([op, class_names[i]])
                if(len(pp) == batch_size):
                    break
    
        np = DataProcessor.get_neg_pairs(pp, operators, objects)
        accLoss = 0
        print(op)
        for i in range(len(imgs)):
            for j in np[i]:               
                
                
                # Img, Class, Op, Neg Class, Neg Op
                loss = model.train_forward(imgs[i], pp[i][1], pp[i][0], j[1], j[0])
                #print(pp[i][1], pp[i][0], j[1], j[0], float(loss))
                accLoss = accLoss + loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        print(accLoss)
    print("----------")



te = 1
for test_epoch in range(0,te):
    batch_size = 3
    loss = 0
    for op in operators: # for each operator...
        class_names = image_testsets[op].classes
        imgs = []
        pp = []
        np = []
        
        while len(imgs) < batch_size:
            inputs, classes = next(iter(testloaders[op]))  # Gather a set of images and classes from them
            print(classes)
            for i in inputs:
                imgs.append(Variable(feat_extractor(i.unsqueeze_(0))))
                if(len(imgs) == batch_size):
                    break
            for i in classes:
                pp.append([op, class_names[i]])
                if(len(pp) == batch_size):
                    break
                
        accLoss = 0
        print(op)
        atAcc, obAcc, bAcc = 0, 0, 0
        for i in range(len(imgs)):            
            a, o, b = model.val_forwardP(imgs[i], pp[i][1], pp[i][0])
            atAcc = atAcc + a/(len(imgs)*te)
            obAcc = obAcc + o/(len(imgs)*te)
            bAcc = bAcc + b/(len(imgs)*te)
                
print("Attribute Accuracy: ", atAcc)
print("Object Accuracy: ", obAcc)
print("Both Accuracy: ", bAcc)


#model.val_forward(imgs[i], pp[i][1],  pp[i][0])

    
    
    
    
    
    
    
    
    
    
    