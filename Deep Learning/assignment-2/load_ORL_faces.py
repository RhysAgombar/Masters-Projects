import numpy as np
import matplotlib.pyplot as pl

# load data
def load_orl():
    data = np.load('ORL_faces.npz')
    trainX = data['trainX']
    trainY = data['trainY']
    testX = data['testX']
    testY = data['testY']
    return trainX, trainY, testX, testY

def sanity_check(trainX):
    fig = pl.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    sanity_check = trainX[0]
    sanity_check = sanity_check.reshape(112,92)
    ax1.imshow(sanity_check, cmap='gray')

    sanity_check = trainX[10]
    sanity_check = sanity_check.reshape(112,92)
    ax2.imshow(sanity_check, cmap='gray')

    sanity_check = trainX[20]
    sanity_check = sanity_check.reshape(112,92)
    ax3.imshow(sanity_check, cmap='gray')

    sanity_check = trainX[30]
    sanity_check = sanity_check.reshape(112,92)
    ax4.imshow(sanity_check, cmap='gray')

    sanity_check = trainX[40]
    sanity_check = sanity_check.reshape(112,92)
    ax5.imshow(sanity_check, cmap='gray')

    sanity_check = trainX[50]
    sanity_check = sanity_check.reshape(112,92)
    ax6.imshow(sanity_check, cmap='gray')

    pl.show()
