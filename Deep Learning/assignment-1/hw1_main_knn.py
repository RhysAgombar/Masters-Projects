# -*- coding: utf-8 -*-
"""
Created on

@author: fame
"""
from load_mnist import *
import hw1_knn  as mlBasics
import numpy as np
from sklearn.metrics import confusion_matrix

#######
## 2a)
#######
u_bound = 10
X_train, y_train = load_mnist('training', range(0,u_bound))
X_test, y_test = load_mnist('testing', range(0,u_bound))


#######
## 2b)
#######
print("############## 2b ##############")

training_set = []
training_labels = []
n = 100
for j in range(0,u_bound):
    hld = np.array(list(np.where(y_train == np.float64(j)))).flatten()
    training_set = training_set + list(np.random.choice(hld, size=n, replace=False))
    training_labels = training_labels + [np.float64(j)] * n

training_set = np.array(training_set)
training_labels = np.array(training_labels)

p = np.random.permutation(len(training_set)) ## This scrambles the training set/labels, which would otherwise be in order
training_set = training_set[p] ## this is a list of indices
training_labels = training_labels[p]  ## this is a list of generated labels

X_train = X_train[training_set,:,:]
y_train = np.array(training_labels)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

dists =  mlBasics.compute_euclidean_distances(X_train,X_test)
y_test_pred_k1 = mlBasics.predict_labels(dists, y_train, k=1)

print '{0:0.02f}'.format(  np.mean(y_test_pred_k1==y_test)*100), "of test examples classified correctly (k = 1)."

y_test_pred_k5 = mlBasics.predict_labels(dists, y_train, k=5)
classifications = np.array(list(map(lambda y: np.argmax(np.bincount(y.astype(np.int64))).astype(np.float64), y_test_pred_k5)))
## For each row in y_test_pred, take the maximum value of a bin count to find the mode of the top k neighbors

print '{0:0.02f}'.format(  np.mean(classifications==y_test)*100), "of test examples classified correctly (k = 5)."

CF_Size = 10

print("Labels: ", y_test[:CF_Size])
print("K=1 Results: ", y_test_pred_k1[:CF_Size])
print("K=5 Results: ", classifications[:CF_Size])

print("K=1 Confusion Matrix: ")
print(confusion_matrix(y_test[:CF_Size], y_test_pred_k1[:CF_Size]))
print("K=5 Confusion Matrix: ")
print(confusion_matrix(y_test[:CF_Size], classifications[:CF_Size]))


#######
## 2c)
#######
print("############## 2c ##############")
def five_fold_CV(dataset, labels, k):
    segments = []
    segment_labels = []

    fold = 5
    seg_size = dataset.shape[0]/fold
    for i in range(1,fold+1):
        segments = segments + [dataset[(i-1)*seg_size:i*seg_size]]
        segment_labels = segment_labels + [labels[(i-1)*seg_size:i*seg_size]]

    acc = []
    for i in range(0,fold):
        tSet = np.empty((0,784), int)
        for j in range(0,fold):
            if (j != i):
                tSet = np.vstack((tSet, segments[i]))

        dists =  mlBasics.compute_euclidean_distances(tSet, segments[i])
        test_pred = mlBasics.predict_labels(dists, segment_labels[i], k)

        if(k == 1):
            acc.append(np.mean(test_pred==segment_labels[i])*100)
        else:
            classifications = np.array(list(map(lambda y: np.argmax(np.bincount(y.astype(np.int64))).astype(np.float64), test_pred)))
            acc.append(np.mean(classifications==segment_labels[i])*100)

    #print("Accuracies from 5-fold CV: ", acc)
    return np.mean(acc)

acc = []
for i in range(1,16):
    acc.append("K = " + str(i) + ": " + str(five_fold_CV(X_train,y_train,k = i)))
print(acc)

#######
## 2d)
#######
print("############## 2d ##############")

u_bound = 10
X_train, y_train = load_mnist('training', range(0,u_bound))
X_test, y_test = load_mnist('testing', range(0,u_bound))

size = 30000
X_train, y_train, X_test, y_test = X_train[:size], y_train[:size], X_test[:size], y_test[:size] # For some reason, it crashes my pc if I use the entire dataset, but as you can see, it still works for a large amount of data

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

dists =  mlBasics.compute_euclidean_distances(X_train,X_test)
y_test_pred_k1 = mlBasics.predict_labels(dists, y_train, k=1)

acc = []
for i in [1,7]:
    acc.append("K = " + str(i) + ": " + str(five_fold_CV(X_train,y_train,k = i)))
print(acc)
