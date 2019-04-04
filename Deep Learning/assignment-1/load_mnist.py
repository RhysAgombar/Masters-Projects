
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:40:10 2016

Taken from https://github.com/tqchen/ML-SGHMC
"""

import os, struct
from array import array as pyarray
from numpy import  array, zeros
from random import shuffle
import numpy as np

def load_mnist(dataset="training", digits=np.arange(10), path='.', random=False, perDigitNum=100):
    """
    Loads MNIST files into 3D numpy arrays 
    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    counter = [0 for k in range(0, 10)] # initialize a counter for digit 0 ~ 9
    N = len(ind)

    images = zeros((N, rows, cols) )
    labels = zeros((N ) )

    shuffle(ind) # suffle the index of images
    for i in range(len(ind)):
        if random == False:
            images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]
        else:
            # check if the couter[digit] exceed perDigitNum
            if counter[lbl[ind[i]]] >= perDigitNum:
                continue
            else:
                images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
                labels[i] = lbl[ind[i]]
                counter[lbl[ind[i]]] += 1

    return images, labels
    
def printArray(arrayToPrint):
    for i in range(len(arrayToPrint)):
        print(arrayToPrint[i])
