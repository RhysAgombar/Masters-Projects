import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)  # if you do not have latex installed simply uncomment this line + line 75


def load_data():
    """ loads the data for this task
    :return:
    """
    fpath = 'images/ball.png'
    radius = 70
    Im = cv2.imread(fpath, 0).astype('float32')/255  # 0 .. 1

    # we resize the image to speed-up the level set method
    Im = cv2.resize(Im, dsize=(0, 0), fx=0.5, fy=0.5)

    height, width = Im.shape

    centre = (width // 2, height // 2)
    Y, X = np.ogrid[:height, :width]
    phi = radius - np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return Im, phi


def get_contour(phi):
    """ get all points on the contour
    :param phi:
    :return: [(x, y), (x, y), ....]  points on contour
    """
    eps = 1
    A = (phi > -eps) * 1
    B = (phi < eps) * 1
    D = (A - B).astype(np.int32)
    D = (D == 0) * 1
    Y, X = np.nonzero(D)
    return np.array([X, Y]).transpose()

# ===========================================
# RUNNING
# ===========================================
def grad(x):
    return np.array(np.gradient(x))

def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))

def stopping_fun(x):
    return 1. / (1. + np.sqrt(np.sum(np.square((np.array(np.gradient(x)))), axis=0)))



if __name__ == '__main__':

    n_steps = 20000
    plot_every_n_step = 100

    dt = 1

    Im, phi = load_data()
    F = stopping_fun(Im) # some function

    #phi = np.multiply(phi, -1)

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)


    for t in range(n_steps):

        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax1.imshow(Im, cmap='gray')
        ax1.set_title('frame ' + str(t))

        dphi = np.array(np.gradient(phi))
        dphi_norm = np.sqrt(np.sum(dphi**2, axis=0))

        phi = phi - dt * F * dphi_norm ## use - to make it converge. But slides say use +?

        contour = get_contour(phi)

        if len(contour) > 0:
            ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=1)
            ax3.scatter(contour[:, 0], contour[:, 1], color='red', s=1)

        ax2.imshow(phi)
        ax2.set_title(r'$\phi$', fontsize=22)



        plt.pause(0.01)

    plt.show()
