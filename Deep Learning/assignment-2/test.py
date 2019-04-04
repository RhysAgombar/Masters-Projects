import numpy as np
import tensorflow as tf
from load_mnist import *
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Load Mnist Data
u_bound = 10
mnist_input_size = 784 ## 28 * 28
X_train, y_train = load_mnist('training', range(0,u_bound))
X_test, y_test = load_mnist('testing', range(0,u_bound))

# Get and Scramble training set, with n items per category
training_set = []
training_labels = []
n = 5
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

# convert X_train to a flat list (28x28 -> 784x1)
#np.set_printoptions(threshold=np.nan, linewidth=28*6, precision=2)
#print(X_train[0])

X_train = X_train.reshape(X_train.shape[0], mnist_input_size)/255 ## numbers are between 0 and 255, so we need to normalize
#np.set_printoptions(threshold=np.nan, linewidth=28*5 + 5, precision=2)
#print(X_train[0])



#convert y_train to set of one-hot vectors
labels = np.zeros((y_train.shape[0], u_bound))
labels[np.arange(y_train.shape[0]), y_train.astype(int)] = 1
y_train = labels


class N():
    pass
model = N()
model.x = tf.placeholder("float", [None, 784])
model.ref = tf.placeholder("float", [None, 10])

model.nn = 10

model.w1 = tf.Variable(tf.random_normal([784, model.nn])),
model.wout = tf.Variable(tf.random_normal([model.nn, 10]))
model.b1 =  tf.Variable(tf.random_normal([model.nn])),
model.bout = tf.Variable(tf.random_normal([10]))

model.l1 = tf.add(tf.matmul(model.x, tf.reshape(model.w1,[784, model.nn])), model.b1)
model.out = tf.matmul(model.l1, model.wout) + model.bout
model.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.out, labels=model.ref))

model.prediction = tf.equal(tf.argmax(model.out,1), tf.argmax(model.ref,1))
model.accuracy = tf.reduce_mean(tf.cast(model.prediction, tf.float32))


optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(model.cost)
s = tf.Session()
s.run(tf.global_variables_initializer())

tAcc, cvAcc = [], []

images, refs = mnist.train.next_batch(100)


feed = {model.x: X_train, model.ref: y_train }

#feed = {model.x: images, model.ref: refs }
for i in range(0,10):
    cost = s.run([model.cost, optimizer], feed)
    tAcc += [cost]#s.run(model.accuracy, feed)]

#feed = {model.x: images, model.ref: refs }
tAcc += [s.run(model.accuracy, feed)]


print(tAcc)
