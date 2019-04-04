import numpy as np
import tensorflow as tf
from load_mnist import *
import matplotlib.pyplot as pl

# Load Mnist Data
u_bound = 10
#mnist_input_size = 784 ## 28 * 28
Train_X, Train_Y = load_mnist('training', range(0,u_bound))
Test_X, Test_Y = load_mnist('testing', range(0,u_bound))

print(Train_X.shape, Train_Y.shape)
print(Test_X.shape, Test_Y.shape)

def getDataSubset(X_train, y_train, X_test, y_test, n=100,u_bound=10,mnist_input_size=784):
    # Get and Scramble training set, with n items per category
    training_set, test_set = [], []
    training_labels, test_labels = [], []
    for j in range(0,u_bound):
        hld = np.array(list(np.where(y_train == np.float64(j)))).flatten()
        training_set = training_set + list(np.random.choice(hld, size=n, replace=False))
        training_labels = training_labels + [np.float64(j)] * n

        hldT = np.array(list(np.where(y_test == np.float64(j)))).flatten()
        test_set = test_set + list(np.random.choice(hldT, size=n, replace=False))
        test_labels = test_labels + [np.float64(j)] * n

    training_set = np.array(training_set)
    training_labels = np.array(training_labels)

    test_set = np.array(test_set)
    test_labels = np.array(test_labels)


    p = np.random.permutation(len(training_set)) ## This scrambles the training set/labels, which would otherwise be in order
    training_set = training_set[p] ## this is a list of indices
    training_labels = training_labels[p]  ## this is a list of generated labels

    X_train = X_train[training_set,:,:]
    y_train = np.array(training_labels)


    p = np.random.permutation(len(test_set)) ## This scrambles the test set/labels, which would otherwise be in order
    test_set = test_set[p] ## this is a list of indices
    test_labels = test_labels[p]  ## this is a list of generated labels

    X_test = X_test[test_set,:,:]
    y_test = np.array(test_labels)

    # convert X_train to a flat list (28x28 -> 784x1)
    #np.set_printoptions(threshold=np.nan, linewidth=28*6, precision=2) ## code for displaying the transformation from 2d to 1d in a readable way, to make sure it's correct
    #print(X_train[0])

    X_train = X_train.reshape(X_train.shape[0], mnist_input_size)/255 ## numbers are between 0 and 255, so we need to normalize
    X_test = X_test.reshape(X_test.shape[0], mnist_input_size)/255
    #np.set_printoptions(threshold=np.nan, linewidth=28*5 + 5, precision=2)
    #print(X_train[0])

    #convert y_train to set of one-hot vectors
    labels = np.zeros((y_train.shape[0], u_bound))
    labels[np.arange(y_train.shape[0]), y_train.astype(int)] = 1
    y_train = labels

    labelsT = np.zeros((y_test.shape[0], u_bound))
    labelsT[np.arange(y_test.shape[0]), y_test.astype(int)] = 1
    y_test = labelsT

    return X_train, y_train, X_test, y_test

u_bound = 10
mnist_input_size = 784

# Create model
tf.reset_default_graph()
class N():  ## This stuff is not, strictly speaking, necessary, but it does help with organization
    pass
model = N()

model.lin = tf.placeholder("float", [None, mnist_input_size]) #inputs
model.ref = tf.placeholder("float", [None, u_bound]) #labels
model.nn = 100 # number of neurons per layer

model.w1 = tf.Variable(tf.random_normal([mnist_input_size, model.nn])) # weights for hidden layer 1
model.w2 = tf.Variable(tf.random_normal([model.nn, model.nn])) # weights for hidden layer 2
model.wout = tf.Variable(tf.random_normal([model.nn, u_bound])) #weights for output layer

model.b1 =  tf.Variable(tf.random_normal([model.nn])) # bias for layer 1
model.b2 =  tf.Variable(tf.random_normal([model.nn])) # bias for layer 2
model.bout = tf.Variable(tf.random_normal([u_bound])) # bias for output layer

#layers
model.l1 = tf.add(tf.matmul(model.lin,tf.reshape(model.w1,[mnist_input_size,model.nn])),model.b1) ## w*x+b
model.l2 = tf.add(tf.matmul(model.l1,tf.reshape(model.w2,[model.nn,model.nn])),model.b2)
model.out = tf.matmul(model.l2,model.wout) + model.bout #tf.add(tf.matmul(model.l2,tf.reshape(model.wout,[model.nn,u_bound])),model.bout)

#cost function, predictions, accuracy, etc.
model.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.out, labels=model.ref))
model.prediction = tf.equal(tf.argmax(model.out,1), tf.argmax(model.ref,1))
model.accuracy = tf.reduce_mean(tf.cast(model.prediction, tf.float32))

## ALTERNATE SOLUTION, MORE ACCURATE BUT POSSIBLY NOT WHAT IS WANTED?
'''
model.L1 = tf.layers.dense(inputs = model.lin, units = model.nn, activation = tf.sigmoid)
model.L2 = tf.layers.dense(inputs = model.L1, units = model.nn, activation = tf.sigmoid)
model.Output = tf.layers.dense(inputs = model.L2, units = u_bound, activation = None)
model.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = model.ref, logits = model.Output))
model.prediction = tf.equal(tf.argmax(model.Output,1), tf.argmax(model.ref,1))
model.accuracy = tf.reduce_mean(tf.cast(model.prediction, tf.float32))
lr = 0.05
'''

#optimizers
lr = 0.0005 #learning rate
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(model.cost)
s = tf.Session()
s.run(tf.global_variables_initializer())

tAcc = []
epochs = 100
iters = 100
n = 100
for j in range(0, epochs):
    if(j%10 == 0):
        print("Epoch: " + str(j) + " of " + str(epochs) + " epochs.")
    X_train, y_train, X_test, y_test = getDataSubset(Train_X, Train_Y, Test_X, Test_Y, n, u_bound, mnist_input_size) # get new data subset for each epoch
    feed = {model.lin: X_train, model.ref: y_train }
    for i in range(0,iters):
        err = s.run([model.cost, optimizer], feed) # train model using a gradient descent optimizer to minimize the model's cost
    feed = {model.lin: X_test, model.ref: y_test }
    tAcc += [s.run(model.accuracy, feed)] ## check accuracy *without* optimizing, such that the model never learns from the test data

print("Epoch: " + str(epochs) + " of " + str(epochs) + " epochs.")

pl.figure()
pl.plot(np.linspace(0,epochs-1,epochs), tAcc[:], '-')
pl.ylabel('Accuracy')
pl.xlabel('Epochs')
pl.title('Accuracy over Time')
pl.show()
