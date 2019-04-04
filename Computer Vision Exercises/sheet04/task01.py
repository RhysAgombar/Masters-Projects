import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2

### coordinate system for plot snake and load data changed to match that of numpy: (y,x)

def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (y0, x0), (y1, x1), ... (yn, xn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0,:]).reshape((-1, 2))
    ax.plot(V_plt[:,1], V_plt[:,0], color=line, alpha=alpha)
    ax.scatter(V[:,1], V[:,0], color=[fill if x != V.shape[0]-1 else 'b' for x in range(V.shape[0]) ],
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (y, x) in enumerate(V):
            ax.text(x, y, str(i))


def load_data(fpath, radius, shift=(0,0)):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    sh, sw = shift
    n = 30  # number of points
    u = lambda i: radius * np.cos(i) + h / 2 +sh
    v = lambda i: radius * np.sin(i) + w / 2 + sw
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here

# ------------------------
def run(fpath, radius, alpha=2, shift=(0,0)):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    Im2, V = load_data(fpath, radius, shift=shift)
    ##v: array of points

    # Preprocess to get rid of the carpet
    Im = cv2.medianBlur(Im2,7)
    Im[np.where(Im < 75)] = 255

    ##Im: image luminance
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 100

    # ------------------------
    # your implementation here

    ##calculating derivates using a sobel filter
    sx = cv2.Sobel(Im,cv2.CV_64F,1,0,ksize=7)
    sy = cv2.Sobel(Im,cv2.CV_64F,0,1,ksize=7)
    G2 = sx*sx + sy*sy




    U = lambda w: - G2[w[0], w[1]]

    P = lambda w1, w2 : alpha * ( (w1[0]-w2[0])**2 +  (w1[1]-w2[1])**2 )

    K = 5
    N = V.shape[0]



    ##initialize weight matrix
    W = np.zeros((K, N))
    ##adjacency matrix. records coordinates (k) of parent
    A = np.zeros((K,N), dtype=np.uint)


    def get_w_nk(w , ks):
        """
        returns the coordinates of neighbour ks
        """
        k_c_sh = [(0,0),(1,0),(0,1),(0,-1),(-1,0)]

        return (w[0]+k_c_sh[ks][0],w[1]+k_c_sh[ks][1])

    ##checks if coordinate is legal (inside image)
    check_coor = lambda w: 0<=w[0] and w[0]<Im.shape[0] and 0<=w[1] and w[1]<Im.shape[1]

    assert(Im.shape == G2.shape)


    for t in range(n_steps):
        # ------------------------
        # your implementation here

        # ------------------------

        ##calculating weights
        for i in range(N):
            for k_ in range(K):

                W[k_,i] = U(get_w_nk(V[i],k_))
                if i == 0:
                    ##need to take into account distance to the previous point (last in V)
                    # since the actual position for the last point is defined only when it is
                    # processed, assume that it will stay at the same point as before (k=0) and
                    # approximate distance
                    W[k_,i] += P(get_w_nk(V[i],k_), get_w_nk(V[-1],0))


                if i != 0: ## not first point, calculate transitions to it
                    inxList = [x for x in range(K)]

                    ##find min link cost
                    inx = min(inxList, key=lambda x: P(get_w_nk(V[i-1],x), get_w_nk(V[i],k_)) + W[x][i-1]  )
                    W[k_,i] += W[inx][i-1] + P(get_w_nk(V[i-1],inx), get_w_nk(V[i],k_))
                    A[k_, i] = inx

                if i == N-1:
                    inxList = [x for x in range(K)]
                    ##find min link cost

                    inx = min(inxList, key=lambda x: P(get_w_nk(V[0],x), get_w_nk(V[i],k_))  )
                    W[k_,i] += P(get_w_nk(V[0],inx), get_w_nk(V[i],k_))


        #finding k_ which minimizes cost for the last point
        k_min = np.argmin(W[:,N-1])
        V[N-1] = get_w_nk(V[N-1], k_min)
        ##update coordinates
        for i in range(N-2, -1,-1): ## going backwards in V
            k_min = A[k_min, i+1]
            #print("V[%d] was " % i, V[i])
            V[i] = get_w_nk(V[i], k_min)
            #print("V[%d] IS " % i, V[i])

        # ------------------------
        ax.clear()

        ax.imshow(Im2, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, V)
        plt.pause(0.001)

    plt.pause(2)


if __name__ == '__main__':
    run('images/ball.png', radius=140,alpha=2, shift=(0,0))
    run('images/coffee.png', radius=130,alpha=2)
