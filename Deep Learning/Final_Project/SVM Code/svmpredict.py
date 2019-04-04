import pickle
import numpy as np
import math
# from sklearn.ensemble import BaggingClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from os import listdir

svm_checkpoints_dir = "/home/wei/Desktop/final-project/checkpoints/"
dataset_dir = "/home/wei/Desktop/hdd/feature-set/"

def main():
    
    num_class = 57
    epoch = 129
    svm = load_model(svm_checkpoints_dir + "sgdSVC-checkpoint-" + str(num_class) + "-" + str(epoch) + ".pickle")
    X, y = load_dataset(dataset_dir, num_class)

    pred = []
    for i in range(0,X.shape[0]): 
        pred.append(svm.predict(X[i,:].reshape(1,-1)))
        # print("current accuracy: ", accuracy_score(y[0:i+1], pred))
    print("final accuracy: ", accuracy_score(y, pred))
    pickle.dump(svm, open("sgdSVC-acu-" + str(num_class) + "-" + str(epoch) + ".pickle", 'wb'))
        
    print("end")

def load_dataset(path_to_datset, max_number_of_dataset=10):
    files = listdir(path_to_datset)
    features = []
    X = []
    y = []
    for idx, file in enumerate(files):
        if idx > max_number_of_dataset:
            break
        else:
            path = path_to_datset + file
            features.append(flatten_cnn_feat(load_feature(path)))

    p = features[0].reshape((1000, 8192))
    for i in range(0, max_number_of_dataset):
        X.append(features[i].reshape((1000, 8192))) # hard code dimension
        y.append(np.ones(1000, dtype=np.float32)*i)
  
    return np.array(X).reshape(max_number_of_dataset*1000, 8192),np.ravel(np.array(y).reshape(max_number_of_dataset*1000, 1))

def load_feature(path_to_feature):
    """
    load the feature set for training/testing svm
    """
    with open(path_to_feature, 'rb') as pickle_file:
        # with function provide resource management, like try {...} final {...}
        content = pickle.load(pickle_file)
    return np.array(content[1])

def load_model(path_to_model):
    """
    load the feature set for training/testing svm
    """
    with open(path_to_model, 'rb') as pickle_file:
        # with function provide resource management, like try {...} final {...}
        model = pickle.load(pickle_file)
    return model

def flatten_cnn_feat(cnn_feats):
    (feat_num, _, _, _) = cnn_feats.shape
    flatten_features = []
    # flatten the all feature
    for i in range(0, feat_num):
        flatten_features.append(cnn_feats[i].flatten()) 
    return np.array(flatten_features)

main()