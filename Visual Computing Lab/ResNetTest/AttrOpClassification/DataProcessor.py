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

def load_word_embeddings(emb_file, vocab):

    vocab = [v.lower() for v in vocab]

    embeds = {}
    for line in open(emb_file, 'r', encoding="utf8"):
        line = line.strip().split(' ')
        
        key = line[0]
        
        line = [float(i) for i in line[1:len(line)]]
        
        wvec = torch.Tensor(line)#map(float, line[1:]))
        embeds[key] = wvec
        
    embeds = [embeds[k] for k in vocab]
    embeds = torch.stack(embeds)
    print('loaded embeddings', embeds.size())

    return embeds


def get_neg_pairs(pp, operators, objects):
    #np = []
    #for i in pp:
    ls = []
    for j in operators:
        for k in objects:
            if ((j != pp[0]) or (k != pp[1])):
                ls.append([j,k])
            #else:
                #print(j, k)
    #np.append(ls)
    return ls
