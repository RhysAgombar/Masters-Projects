import pickle
import numpy as np
import math
# from sklearn.ensemble import BaggingClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from os import listdir

dataset_dir = "/home/wei/Desktop/hdd/feature-set/"

def main():
    """
    this is the main function for tranning svm
    """
    num_class = 57

    X, y = load_dataset(dataset_dir, num_class)
    
    folds = 5;
    n = math.floor(X.shape[0]/folds)
    X, y = shuffle(X, y, random_state=0)

    train = np.arange(0, (folds-2)*n)
    test = np.arange((folds-2)*n, X.shape[0])
    
    clf = linear_model.SGDClassifier(n_jobs=-1)

    k = 500
    epoch = 250
    iteration = 1000
    mini_batch = math.floor(train.shape[0]/iteration)
    train_plot = []
    valid_plot = []
    for ep in range(0, epoch):
        for it in range(0, iteration-1):
            batch = train[it*mini_batch:(it+1)*mini_batch]
            clf.partial_fit(X[batch], y[batch], classes=np.arange(0,num_class))
            pred_tra = clf.predict(X[train[0:k]])
            pred_val = clf.predict(X[test[0:k]])
            print("train_score: [" + str(ep) + "]" + "[" + str(it) + "]", accuracy_score(y[train[0:k]], pred_tra))
            print("valid_score: [" + str(ep) + "]" + "[" + str(it) + "]", accuracy_score(y[test[0:k]], pred_val))
        train_plot.append(accuracy_score(y[train[0:k]], pred_tra))
        valid_plot.append(accuracy_score(y[test[0:k]], pred_val))
        pickle.dump([train_plot, valid_plot], open("error-plot-checkpoint-" + str(num_class) + "-" + str(ep) + ".pickle", 'wb'))
        pickle.dump(clf, open("sgdSVC-checkpoint-" + str(num_class) + "-" + str(ep) + ".pickle", 'wb'))
    # clf = OneVsRestClassifier(BaggingClassifier(svm.SVC(kernel='linear', probability=True), max_samples=1.0 / num_class, n_estimators=num_class))
    # clf = BaggingClassifier(svm.LinearSVC(), max_samples=1.0/num_estimator, n_estimators=num_estimator, n_jobs=2)
    # clf = svm.SVC()
    # clf.fit(X[train], y[train])
    pickle.dump(clf, open("sgdSVC-" + str(num_class) + "-" + str(epoch) + ".pickle", 'wb'))
    pred = clf.predict(X[test])
    # pickle.dump(pred, open("prediction-" + str(num_class) + ".pickle", 'wb'))
    # # report accuracy of the model prediction
    print("final accuracy: ",accuracy_score(y[test], pred))


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

def flatten_cnn_feat(cnn_feats):
    (feat_num, _, _, _) = cnn_feats.shape
    flatten_features = []
    # flatten the all feature
    for i in range(0, feat_num):
        flatten_features.append(cnn_feats[i].flatten()) 
    return np.array(flatten_features)

main()
