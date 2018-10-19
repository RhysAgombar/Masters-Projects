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
import DataProcessor
from numpy import unravel_index


class MLP(nn.Module): ## The issue is somewhere in here
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True):
        super(MLP, self).__init__()
        mod = []
        for L in range(num_layers-1):
            mod.append(nn.Linear(inp_dim, inp_dim, bias=bias))
            mod.append(nn.ReLU(True))

        mod.append(nn.Linear(inp_dim, out_dim, bias=bias))
        if relu:
            mod.append(nn.ReLU(True))

        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output


class AttrOpModel(nn.Module):
    def __init__(self, data):
        super(AttrOpModel, self).__init__()
        
        self.objects, self.operators = data[0], data[1]
        
        dim = 10 # 300
        self.image_embedder = MLP(512, dim) # 512 image features embedded into 300
        
        self.attr_ops = nn.ParameterList([nn.Parameter(torch.eye(dim)) for _ in range(len(self.operators))])
        self.obj_embedder = nn.Embedding(len(self.objects), dim)     
        
        #pretrained_weight = DataProcessor.load_word_embeddings('glove/glove.6B.300d.txt', self.objects)
        #self.obj_embedder.weight.data.copy_(pretrained_weight)

        #self.pdist_func = F.pairwise_distance

        self.inverse_cache = {}
        
        f = open("train_pairs.txt",'r')
        self.pairs = f.read().strip().split('\n')
        self.pairs = [t.split() for t in self.pairs]
        self.pairs = map(tuple, self.pairs)
        
        self.attrsP, self.objsP = zip(*self.pairs)
        
    def apply_op(self, obj, op):
        dim = 10 # 300
        out = torch.bmm(obj.view(1,1,dim), op.view(1,dim,dim)) 
        out = F.relu(out).view(dim)
        return out
        
        
    def train_forward(self, img, obj_label, pos_op_label, neg_obj, neg_op_label):
        anchor = self.image_embedder(img)
        
        obj_emb = self.obj_embedder(torch.tensor(obj_label))# , dtype=torch.long))
        pos_op = self.attr_ops[pos_op_label]
        positive = self.apply_op(obj_emb, pos_op)


        neg_obj = torch.tensor(self.objects.index(neg_obj))
        neg_obj_emb = self.obj_embedder(neg_obj)#, dtype=torch.long))
        
        neg_op_label = torch.tensor(self.operators.index(neg_op_label))
        neg_op = self.attr_ops[neg_op_label]
        negative = self.apply_op(neg_obj_emb, neg_op)

        #print(positive)
        #print()
        #print(negative)

        loss = F.triplet_margin_loss(anchor, positive, negative, margin=1.5)


        return loss
    
    
    ### NEED TO COMPLETE ###
    ## For printing distance matrices
    def val_forward(self, img, obj_label, attr_label):
        
        img_feat = self.image_embedder(img)
        attrs, objs = self.attrsP, self.objsP
        
        dists = np.zeros((len(set(attrs)),len(set(objs))))
        for i in range(len(set(attrs))):
            for j in range(len(set(objs))):
                obj_rep = self.obj_embedder(torch.tensor(self.objects.index(objs[j]), dtype=torch.long))
                attr_op = self.attr_ops[torch.tensor(self.operators.index(attrs[i]), dtype=torch.long)]
                embedded_reps = self.apply_op(obj_rep, attr_op)
    
               # print(obj_rep)
               # print()
               # print(attr_op)
               # print()
               # print(embedded_reps)
    
                dist = float(torch.dist(img_feat, embedded_reps))
                
                dists[i,j] = dist

        prediction = unravel_index(dists.argmin(), dists.shape)
        
        
        #print("Prediction: ", attrs[prediction[0]], objs[prediction[1]])
        #print("Actual: ", attrs[attr_label], objs[obj_label])
        #print(dists)
        
        atOut, obOut, bOut = 0, 0, 0
        if(attrs[prediction[0]] == attrs[attr_label]):
            atOut = 1
        if(objs[prediction[1]] == objs[obj_label]):
            obOut = 1
        if(objs[prediction[1]] == objs[obj_label] and attrs[prediction[0]] == attrs[attr_label]):
            bOut = 1
            
        return atOut, obOut, bOut
    #####################################################################
    
    
    
    ## For printing distance matrices
    def val_forwardP(self, img, obj_label, attr_label):
        
        img_feat = self.image_embedder(img)
        attrs, objs = self.attrsP, self.objsP ## Fix this
        
        ## need proper attr/op lists
        ## distance between embedding of attr op pairs and 
        
        
        
        attrSize = len(set(attrs)) 
        objSize = len(set(objs))
        dists = np.zeros((attrSize,objSize))
        
        for i in range(attrSize):
            for j in range(objSize):
                obj_rep = self.obj_embedder(torch.tensor(self.objects.index(objs[j]), dtype=torch.long))
                attr_op = self.attr_ops[torch.tensor(self.operators.index(attrs[i]), dtype=torch.long)]
                embedded_reps = self.apply_op(obj_rep, attr_op)
    
                dist = torch.dist(img_feat, embedded_reps)
                
                dists[i,j] = dist

        prediction = unravel_index(dists.argmin(), dists.shape)
        
        
        print("Prediction: ", attrs[prediction[0]], objs[prediction[1]])
        print("Actual: ", attrs[attr_label], objs[obj_label])
        #print(dists)
        
        atOut, obOut, bOut = 0, 0, 0
        if(attrs[prediction[0]] == attrs[attr_label]):
            atOut = 1
        if(objs[prediction[1]] == objs[obj_label]):
            obOut = 1
        if(objs[prediction[1]] == objs[obj_label] and attrs[prediction[0]] == attrs[attr_label]):
            bOut = 1
            
        return atOut, obOut, bOut
    
    
    def print_dists(self, img, obj_label, attr_label):
        img_feat = self.image_embedder(img)
        attrs, objs = self.attrsP, self.objsP
        
        dists = []
        for i in range(len(attrs)):
            obj_rep = self.obj_embedder(torch.tensor(self.objects.index(objs[i]), dtype=torch.long))
            attr_op = self.attr_ops[torch.tensor(self.operators.index(attrs[i]), dtype=torch.long)]
            embedded_reps = self.apply_op(obj_rep, attr_op)

            dist = torch.dist(img_feat, embedded_reps)
            print("Dist from ", objs[i] )
            
            dists.append(dist)
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  