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

def w(x):
    return 1. / (1. + np.sqrt(np.sum(np.square((np.array(np.gradient(x)))), axis=0)))
'''
def compute_dist_map(Im, C):
    mp = np.zeros(Im.shape)

    for y in range(Im.shape[0]):
        for x in range(Im.shape[1]):
            pos = np.argmin(np.linalg.norm(C-[x,y], axis=1))
            dist = cv2.pointPolygonTest(C,(x,y),True)
            if (dist < 0):
                mp[y,x] = np.linalg.norm(C[pos]-[x,y])
            else:
                mp[y,x] = -1* np.linalg.norm(C[pos]-[x,y])

    return mp
'''
def get_derivs(img):
    ## Change this from per point to over entire map?
    ## Return gradient grid instead?
    dx,dy,dxx,dyy,dxy = np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape)

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            dx[i,j] = (1/2) * (img[i+1,j] - img[i-1,j]) # add abs?
            dy[i,j] = (1/2) * (img[i,j+1] - img[i,j-1]) # add abs?
            dxx[i,j] = img[i+1,j] - 2*(img[i,j] + img[i-1,j])
            dyy[i,j] = img[i,j+1] - 2*(img[i,j] + img[i,j-1])
            dxy[i,j] = (1/4) * (img[i+1,j+1] - img[i+1,j-1] - img[i-1,j+1] + img[i-1,j-1])


    #dx = (1/2) * (Im[pos[0]+1,pos[1]] - Im[pos[0]-1,pos[1]])
    #dy = (1/2) * (Im[pos[0],pos[1]+1] - Im[pos[0],pos[1]-1])
    #dxx = Im[pos[0]+1,pos[1]] - 2*Im[pos[0],pos[1]] + Im[pos[0]-1,pos[1]]
    #dyy = Im[pos[0],pos[1]+1] - 2*Im[pos[0],pos[1]] + Im[pos[0],pos[1]-1]
    #dxy = (1/4)*(Im[pos[0]+1,pos[1]+1] - Im[pos[0]+1,pos[1]-1] - Im[pos[0]-1,pos[1]+1] + Im[pos[0]-1,pos[1]-1])

    return dx, dy, dxx, dyy, dxy


if __name__ == '__main__':

    n_steps = 20000
    plot_every_n_step = 100

    Im, phi = load_data()

    #phi = np.multiply(phi, -1)

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    #contour = get_contour(phi)

    #dx, dy, dxx, dyy, dxy = get_derivs(phi)
    e = 10e-4


    for t in range(n_steps):

#        if t % plot_every_n_step == 0:
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax1.imshow(Im, cmap='gray')
        ax1.set_title('frame ' + str(t))

        dx, dy, dxx, dyy, dxy = get_derivs(phi)

        wf = w(phi)#Im
        tf = 1/(4*np.amax(wf))

        #print(wf.shape)
        #print(phi.shape)
        #phi = phi + (dxx*(dy**2) - 2*dx*dy*dxy + dyy*(dx**2))/(dx**2 + dy**2 + e)
        #phi = phi + (dxx*(np.power(dy,2)) - 2*dx*dy*dxy + dyy*(np.power(dx,2)))/(np.power(dx,2) + np.power(dy,2) + e)

        ## Function
        phi = phi + tf*wf*(dxx*(np.power(dy,2)) - 2*dx*dy*dxy + dyy*(np.power(dx,2)))/(np.power(dx,2) + np.power(dy,2) + e)

        print("----")
        print(phi)

        #phi = np.multiply(phi, -1)
        contour = get_contour(phi)
        #dist = compute_dist_map(Im, contour)


        ax4.imshow(dx)
        ax4.set_title("dx", fontsize=22)

        ax3.imshow(dy)
        ax3.set_title("dy", fontsize=22)

        #for i in range(0,10):#contour.shape[0]):
        #    dx, dy, dxx, dyy, dxy = get_derivs(dist, contour[i])
        #    contour[i][0] = float(contour[i][0]) + float(dx)
        #    contour[i][1] = float(contour[i][1]) + float(dy)
        #    print(contour[i])





        #contour = np.append(contour, [[25,37],[30,37],[35,37], [40,37]], axis=0)

        if len(contour) > 0:
            ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=1)
            ax3.scatter(contour[:, 0], contour[:, 1], color='red', s=1)

        ax2.imshow(phi)
        ax2.set_title(r'$\phi$', fontsize=22)



        plt.pause(0.01)

    plt.show()
