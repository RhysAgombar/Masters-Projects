# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""

import numpy as np 
import matplotlib.pyplot as plt

 

def predict(X,W,b):  
    """
    implement the function h(x, W, b) here  
    X: N-by-D array of training data 
    W: D dimensional array of weights
    b: scalar bias

    Should return a N dimensional array  
    """
    return np.squeeze(sigmoid(np.dot(X,W[:,np.newaxis])+b))
 
def sigmoid(a): 
    """
    implement the sigmoid here
    a: N dimensional numpy array

    Should return a N dimensional array  
    """
    return 1/(1+np.exp(-a))
    

def l2loss(X,y,W,b):  
    """
    implement the L2 loss function
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias

    Should return three variables: (i) the l2 loss: scalar, (ii) the gradient with respect to W, (iii) the gradient with respect to b
     """
    h = sigmoid(np.dot(X,W[:,np.newaxis])+b)
    y = y[:, np.newaxis]
    # Calulate common coefficient in dLdw & dLdb
    c = np.multiply(y - h, -h)
    np.multiply(c, 1-h, c)
    # print(norm2(np.squeeze(2 * np.dot(X.T,c))))
    # print(h)
    return np.sum(np.square(y-h)), np.squeeze(2 * np.dot(c.T,X)), 2 * np.sum(c)
    


def train(X,y,W,b, num_iters=1000, eta=0.1):  
    """
    implement the gradient descent here
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias
    num_iters: (optional) number of steps to take when optimizing
    eta: (optional)  the stepsize for the gradient descent

    Should return the final values of W and b    
     """

    L, dLdw, dLdb = l2loss(X,y,W,b)
    epochs = range(0, num_iters)
    losses = []
    for epoch in epochs:
        W = W - dLdw * eta
        b = b - dLdb * eta
        L, dLdw, dLdb = l2loss(X,y,W,b)
        losses.append(L)
        print("At epoch(" + str(epoch) + ") :" + "L2 = " + str(L) + ", ||dLdw||_2 = " + str(norm2(dLdw)))
    
    plt.plot(epochs, losses)
    plt.xlabel("epochs")
    plt.ylabel("L_2 loss")
    plt.show()
    return W, b
 
def norm2(v):
    return np.sum(np.square(v))