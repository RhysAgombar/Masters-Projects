# -*- coding: utf-8 -*-
"""
Created on

@author: fame
"""

import numpy as np

def compute_euclidean_distances( X, Y ) :
    """
    Compute the Euclidean distance between two matricess X and Y
    Input:
    X: N-by-D numpy array
    Y: M-by-D numpy array

    Should return dist: M-by-N numpy array
    """

    dist = np.sqrt(np.sum(np.square(X)[:,np.newaxis,:], axis=2) - 2 * X.dot(Y.T) + np.sum(np.square(Y), axis=1)).T

    return dist



def predict_labels( dists, labels, k=1):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array
    labels: is a N dimensional numpy array

    Should return  pred_labels: M dimensional numpy array
    """

    matches = np.argpartition(dists, k, axis=1)[:, :k] # The matches that are returned are not in order, however ordering is not necessary.

    if (k == 1):
        matches = matches.flatten()

    matches = np.remainder(matches,labels.shape[0])

    pred_labels = labels[matches]

    return pred_labels
