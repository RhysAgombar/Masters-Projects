import numpy as np
import tensorflow as tf
from load_ORL_faces import *
import matplotlib.pyplot as pl
import random
from scipy.ndimage.filters import gaussian_filter1d

trainX, trainY, testX, testY = load_orl();
trainY = trainY.reshape(trainY.shape[0],1)
testY = testY.reshape(testY.shape[0],1)

testX = testX / 255
trainX = trainX / 255

trainX2 = np.copy(trainX)
for i in range(trainX.shape[0]):
    trainX2[i] = trainX2[i] + np.random.uniform(0, 0.3, trainX.shape[1]) ## Add random noise to a batch of images

trainX3 = np.copy(trainX)
for i in range(trainX.shape[0]):
    trainX3[i] = gaussian_filter1d(trainX3[i], sigma=3) ## Add gaussian blur to a batch of images

sanity_check(trainX)
sanity_check(trainX2)
sanity_check(trainX3)

trainY2 = np.copy(trainY)
trainX = np.append(trainX, trainX2, axis=0) # adding modified training data to original training data. This helps the system generalize and pushes accuracy from ~50% to ~60%
trainX = np.append(trainX, trainX3, axis=0)
trainY = np.append(trainY, trainY2, axis=0)
trainY = np.append(trainY, trainY2, axis=0)

u_bound = 20

labels = np.zeros((trainY.shape[0], u_bound))
for i in range(trainY.shape[0]):
    labels[i, trainY[i]] = 1
trainY = labels


labels = np.zeros((testY.shape[0], u_bound))
for i in range(testY.shape[0]):
    labels[i, testY[i]] = 1
testY = labels


orl_input_size = trainX.shape[1]

tf.reset_default_graph()
class N():  ## This stuff is not, strictly speaking, necessary, but it does help with organization
    pass
model = N()

model.lin = tf.placeholder("float", [None, orl_input_size]) #inputs
model.ref = tf.placeholder("float", [None, u_bound]) #labels
model.nn = 256 # number of neurons per layer


model.w1 = tf.Variable(tf.random_normal([orl_input_size, model.nn])) # weights for hidden layer 1
model.w2 = tf.Variable(tf.random_normal([model.nn, model.nn])) # weights for hidden layer 2
model.w3 = tf.Variable(tf.random_normal([model.nn, model.nn])) # weights for hidden layer 3
model.wout = tf.Variable(tf.random_normal([model.nn, u_bound])) #weights for output layer

model.b1 =  tf.Variable(tf.random_normal([model.nn])) # bias for layer 1
model.b2 =  tf.Variable(tf.random_normal([model.nn])) # bias for layer 2
model.b3 =  tf.Variable(tf.random_normal([model.nn])) # bias for layer 2
model.bout = tf.Variable(tf.random_normal([u_bound])) # bias for output layer

#layers
model.l1 = tf.add(tf.matmul(model.lin,tf.reshape(model.w1,[orl_input_size,model.nn])),model.b1) ## w*x+b
model.l2 = tf.add(tf.matmul(model.l1,tf.reshape(model.w2,[model.nn,model.nn])),model.b2)
model.l3 = tf.add(tf.matmul(model.l2,tf.reshape(model.w3,[model.nn,model.nn])),model.b3)
model.out = tf.matmul(model.l3,model.wout) + model.bout #tf.add(tf.matmul(model.l2,tf.reshape(model.wout,[model.nn,u_bound])),model.bout)

#cost function, predictions, accuracy, etc.
model.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.out, labels=model.ref))
model.prediction = tf.equal(tf.argmax(model.out,1), tf.argmax(model.ref,1))
model.accuracy = tf.reduce_mean(tf.cast(model.prediction, tf.float32))


lr = 1e-6 #learning rate
'''

model.L1 = tf.layers.dense(inputs = model.lin, units = model.nn, activation = tf.sigmoid)
model.L2 = tf.layers.dense(inputs = model.L1, units = model.nn, activation = tf.sigmoid)
model.L3 = tf.layers.dense(inputs = model.L2, units = model.nn, activation = tf.sigmoid)
model.Output = tf.layers.dense(inputs = model.L3, units = u_bound, activation = None)
model.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = model.ref, logits = model.Output))
model.prediction = tf.equal(tf.argmax(model.Output,1), tf.argmax(model.ref,1))
model.accuracy = tf.reduce_mean(tf.cast(model.prediction, tf.float32))
lr = 1e-2 #learning rate
'''


optimizer = tf.train.GradientDescentOptimizer(lr).minimize(model.cost)
s = tf.Session()
s.run(tf.global_variables_initializer())

iters = 100
feed = {model.lin: trainX, model.ref: trainY }
feed2 = {model.lin: testX, model.ref: testY }
acc = []
err = []
ln = 50
for j in range(0,ln):
    print("Iteration: ", str(j*iters), " of ", str(ln*iters))
    for i in range(0,iters):
        hld = s.run([model.cost, optimizer], feed)
    err += [s.run([model.cost, optimizer], feed)]
    acc += [s.run(model.accuracy, feed2)]

    print(err[j])
    print(acc[j])

pl.plot(acc)
pl.ylabel('Accuracy')
pl.xlabel('Hundred Iterations')
pl.title('Accuracy over Time')
pl.show()


















#asda
