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
        self.training = True

        #layers
        self.c1 = tf.layers.conv2d(inputs=self.lin, filters=self.nf, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
        self.p1 = tf.layers.max_pooling2d(inputs=self.c1, pool_size=[2,2], strides=2)
        self.c2 = tf.layers.conv2d(inputs=self.p1, filters=self.nf*2, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
        self.p2 = tf.layers.max_pooling2d(inputs=self.c2, pool_size=[2,2], strides=2)
        self.c3 = tf.layers.conv2d(inputs=self.p2, filters=self.nf*3, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
        self.p3 = tf.layers.max_pooling2d(inputs=self.c2, pool_size=[2,2], strides=2)

        #self.s = tf.shape(self.p3)

        self.flat = tf.reshape(self.p3, [-1, 23*28*48]) #90*110*48])

        self.dl = tf.layers.dense(inputs=self.flat, units=512, activation=tf.nn.relu)
        self.dr = tf.layers.dropout(self.dl, rate=0.4, training=self.training)


        self.out = tf.layers.dense(self.dr, units=u_bound, activation=tf.nn.relu)


        #cost function, predictions, accuracy, etc.
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out, labels=self.ref))
        self.prediction = tf.equal(tf.argmax(self.out,1), tf.argmax(self.ref,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))
        self.visualize = self.c1

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
            model.training = True
            for i in range(0,iters):
                hld = s.run([model.cost, optimizer], train_feed)
            err += [s.run([model.cost, optimizer], train_feed)[0]]
            model.training = False
            acc += [s.run(model.accuracy, test_feed)]
            print("Error: ", err[j], "Accuracy: ", acc[j])

        vis_feed = feeds[2]
        vis = s.run(model.visualize, vis_feed)
        return err, acc, vis


lr = 1e-4 #learning rate
err_m, acc_m = [], []

model = CNN()
train_feed = {model.lin: trainX, model.ref: trainY }
test_feed = {model.lin: testX, model.ref: testY }
vis_feed = {model.lin: testX[0].reshape((1, 92, 112, 1)), model.ref: testY[0].reshape((1, 20)) }
err_l, acc_1, vis_l = model.train(lr, iters=10, epochs=30, feeds=[train_feed, test_feed, vis_feed])
#pl.plot(err_l)

#pl.legend(loc='lower right')
#pl.ylabel('Loss')
#pl.xlabel('Per 10 Iterations')
#pl.title('Loss over Time')
#pl.show()


#w=92
#h=112
fig=pl.figure(figsize=(5, 5))
for i in range(0, 24):
    img = vis_l[0]
    img = img[:,:,i]
    img = img.reshape((112,92))
    fig.add_subplot(5, 5, i+1)
    pl.imshow(img)

pl.show()
print(acc_m)
