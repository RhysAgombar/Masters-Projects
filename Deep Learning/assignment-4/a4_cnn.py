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

'''
trainX2 = np.copy(trainX)
for i in range(trainX.shape[0]):
    trainX2[i] = trainX2[i] + np.random.uniform(0, 0.3, trainX.shape[1]) ## Add random noise to a batch of images

trainX3 = np.copy(trainX)
for i in range(trainX.shape[0]):
    trainX3[i] = gaussian_filter1d(trainX3[i], sigma=3) ## Add gaussian blur to a batch of images
'''

#sanity_check(trainX)
#sanity_check(trainX2)
#sanity_check(trainX3)

'''
trainY2 = np.copy(trainY)
trainX = np.append(trainX, trainX2, axis=0) # adding modified training data to original training data. This helps the system generalize and pushes accuracy from ~50% to ~60%
trainX = np.append(trainX, trainX3, axis=0)

trainY = np.append(trainY, trainY2, axis=0)
trainY = np.append(trainY, trainY2, axis=0)
'''

trainX = trainX.reshape((trainX.shape[0],92,112))
trainX = trainX[:,:,:,np.newaxis]
testX = testX.reshape((testX.shape[0],92,112))
testX = testX[:,:,:,np.newaxis]


u_bound = 20

labels = np.zeros((trainY.shape[0], u_bound))
for i in range(trainY.shape[0]):
    labels[i, trainY[i]] = 1
trainY = labels


labels = np.zeros((testY.shape[0], u_bound))
for i in range(testY.shape[0]):
    labels[i, testY[i]] = 1
testY = labels


#orl_input_size = trainX.shape[1]

class CNN():  ## This stuff is not, strictly speaking, necessary, but it does help with organization
    def __init__(self):
        tf.reset_default_graph()
        self.lin = tf.placeholder("float", [None,92,112,1]) #inputs
        self.ref = tf.placeholder("float", [None, u_bound]) #labels
        self.nf = 24 #12x4, same as the LM Filter Bank for texture analysis #number of filters per layer

        #layers
        self.c1 = tf.layers.conv2d(inputs=self.lin, filters=self.nf, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
        self.p1 = tf.layers.max_pooling2d(inputs=self.c1, pool_size=[2,2], strides=2)
        self.c2 = tf.layers.conv2d(inputs=self.p1, filters=self.nf*2, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
        self.p2 = tf.layers.max_pooling2d(inputs=self.c2, pool_size=[2,2], strides=2)
        self.c3 = tf.layers.conv2d(inputs=self.p2, filters=self.nf*3, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
        self.p3 = tf.layers.max_pooling2d(inputs=self.c2, pool_size=[2,2], strides=2)

        #self.s = tf.shape(self.p3)

        self.flat = tf.reshape(self.p3, [-1, 23*28*48]) #90*110*48])

        self.out = tf.layers.dense(self.flat, units=u_bound, activation=tf.nn.relu)

        #self.dr = tf.layers.dropout(self.out, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)


        #cost function, predictions, accuracy, etc.
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out, labels=self.ref))
        self.prediction = tf.equal(tf.argmax(self.out,1), tf.argmax(self.ref,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))

    def train(self, lr, iters, epochs, feeds):
        optimizer = tf.train.AdamOptimizer(lr).minimize(self.cost)
        s = tf.Session()
        s.run(tf.global_variables_initializer())

        train_feed = feeds[0]
        test_feed = feeds[1]

        acc = []
        err = []
        for j in range(0,epochs):
            print("Iteration: ", str(j*iters), " of ", str(epochs*iters))
            for i in range(0,iters):
                hld = s.run([model.cost, optimizer], train_feed)
            err += [s.run([model.cost, optimizer], train_feed)[0]]
            acc += [s.run(model.accuracy, test_feed)]
            print("Error: ", err[j], "Accuracy: ", acc[j])

        return err, acc


lr = 1e-6 #learning rate
err_m, acc_m = [], []
for i in range(1,6):
    model = CNN()
    train_feed = {model.lin: trainX, model.ref: trainY }
    test_feed = {model.lin: testX, model.ref: testY }
    lr = (10**i)*1e-7 #learning rate
    err_l, acc_1 = model.train(lr, iters=10, epochs=30, feeds=[train_feed, test_feed])
    err_m += [err_l]
    acc_m += [acc_1]
    l = '1e-'+str(7-i)
    pl.plot(err_m[i-1], label=l)

pl.legend(loc='lower right')
pl.ylabel('Loss')
pl.xlabel('Per 10 Iterations')
pl.title('Loss over Time')
pl.show()

pl.savefig('error_over_time_base_cnn.png')
print(acc_m)
