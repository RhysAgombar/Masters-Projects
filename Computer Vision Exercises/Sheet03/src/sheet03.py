import numpy as np
import cv2 as cv
import random


##############################################
#     Task 1        ##########################
##############################################


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/shapes.png')
    imgBW = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    edges = cv.Canny(img,50,100)

    cv.imshow("1a - Edges", edges)

    lines = cv.HoughLines(edges, 1, np.pi/2, 50) #np.pi/2
    for i in range(0, len(lines)):
        print(lines[i][0][0],lines[i][0][1])
        r = lines[i][0][0]
        t = lines[i][0][1]
        a = np.cos(t)
        b = np.sin(t)
        x0 = a * r
        y0 = b * r
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(img, pt1, pt2, (0,0,255))


    print("---")

    cv.imshow("1a - HLines",img)



def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    accumulator = np.zeros((int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)))
    detected_lines = []

    locs = np.where(img_edges != 0)
    locs = np.array(list(zip(locs[0],locs[1])))

    for x,y in locs:
        x_1 = x / d_resolution # Scale by resolution... maybe.
        y_1 = y / d_resolution
        for t in range(0, int(180 / theta_step_sz)):
            theta = t  # - int(180 / theta_step_sz)/2)
            d = int(x_1 * np.cos(theta) - y_1 * np.sin(theta)) # Something is wrong with this, but I'm not certain what. It's the same equation as what's in the slides...
            accumulator[t,d] += 1


    accumulator = accumulator.astype(np.uint8)
    accumulator[np.where(accumulator < threshold)] = 0

    detected_lines = np.where(accumulator > 0)
    detected_lines = np.array(list(zip(detected_lines[0],detected_lines[1])))

    return detected_lines, accumulator

def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) # convert the image into grayscale
    edges = cv.Canny(img,50,100) # detect the edges

    img2 = np.copy(img)

    detected_lines, accumulator = myHoughLines(edges, 1, 1, 50)

    print(detected_lines)

    cv.imshow("1b - Accumulator", accumulator)


    for i in range(0, len(detected_lines)):
        t = float(detected_lines[i][0])
        r = float(detected_lines[i][1])

        a = np.cos(t)
        b = np.sin(t)

        x0 = a * r
        y0 = b * r + 85 ## only to show the line furthest left. I have no idea why the scaling is off.
        pt1 = (int(y0 + 1000*(a)), int(x0 + 1000*(-b)))
        pt2 = (int(y0 - 1000*(a)), int(x0 - 1000*(-b)))
        #pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        #pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(img, pt1, pt2, (0,0,255))


    cv.imshow("1b - Lines", img)


##############################################
#     Task 2        ##########################
##############################################


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) # convert the image into grayscale
    edges = cv.Canny(img_gray,50,100) # detect the edges
    theta_res = 1 # set the resolution of theta
    d_res = 1 # set the distance resolution
    _, accumulator = myHoughLines(edges, d_res, theta_res, 10)

    cv.imshow("2 - Accumulator", accumulator)
    cv.imshow("2 - Edges", edges)

    cent = (np.random.rand() * accumulator.shape[0], np.random.rand() * accumulator.shape[1]) #(accumulator.shape[0]/2,accumulator.shape[1]/2)
    dist = 1e9

    locs = np.where(accumulator > 0)
    locs = np.array(list(zip(locs[0],locs[1])))

    while(dist>0.5):
        wiRange = []
        rng = 10
        for i in range(0, len(locs)):
            if (np.linalg.norm(locs[i]-cent) > rng):
                wiRange.append(list(locs[i]))

        nm = np.array([0.0,0.0])
        for i in range(0,len(wiRange)):
            nm[0] += wiRange[i][0]/len(wiRange)
            nm[1] += wiRange[i][1]/len(wiRange)

        dist = np.linalg.norm(cent-nm)
        print(dist)
        print(cent)
        print(nm)
        print("----")
        cent = np.copy(nm)

    print(int(cent[0]),int(cent[1]))

    accumulator[int(cent[0])-5:int(cent[0])+5,int(cent[1])-5:int(cent[1])+5] = 255
    cv.imshow("2 - Accumulator with square for middle location",accumulator)


    t = float(cent[0])
    r = float(cent[1])

    a = np.cos(t)
    b = np.sin(t)

    x0 = a * r
    y0 = b * r + 85 ## only to show the line furthest left. I have no idea why the scaling is off.
    pt1 = (int(y0 + 1000*(a)), int(x0 + 1000*(-b)))
    pt2 = (int(y0 - 1000*(a)), int(x0 - 1000*(-b)))
    #pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv.line(img, pt1, pt2, (0,0,255))

    cv.imshow("2 - Image with line", img)



##############################################
#     Task 3        ##########################
##############################################



def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster (each line one point, each column one attribute)
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """

    ##limit on itearations
    max_iter = 20
    ##treshold centroid shift: threshold to what is considered a shift in centroid position
    threshold = 0.1


    #centers: k lines and d columns
    centers = np.zeros((k, data.shape[1]))

    ##index: maps points to clusters
    index = np.zeros(data.shape[0], dtype=np.uint8)
    clusters = [[] for i in range(k)]

    dimensions = data.shape[1] ## dimensions in data

    # initialize centers using some random points from data
    # ....
    max_val = np.max(data, axis=0)
    min_val = np.min(data, axis=0)

    for k_i in range(k):
        for d in range(dimensions):
            centers[k_i][d] = random.randrange(min_val[d], max_val[d])

    convergence = False
    iterationNo = 0
    while not convergence:
        # assign each point to the cluster of closest center
        # ...
        for pinx, point in enumerate(data):
            diff = centers - point
            norm2 = np.linalg.norm(diff, axis = 1)
            centerIndex = norm2.argmin()
            index[pinx] = centerIndex

        convergence = True
        # update clusters' centers and check for convergence
        # ...
        for centerInx in range(k):
            pointsInx = np.where(index == centerInx)

            if not (np.any(np.isnan(pointsInx[0])) or pointsInx[0].size == 0) :#if there is no point assigned to cluster does nothing
                avg_point = data[pointsInx].mean(axis= 0) ##calculate the mean value for each dimension
                centroid_shift = np.linalg.norm(avg_point - centers[centerInx])
                centers[centerInx] = avg_point
                ##stop conditions
                ##check for distance of centroid shift. if all centroids shift by less than threshold, stop.
                if centroid_shift > threshold:
                    convergence = False


                if iterationNo > max_iter:
                    convergence = True

        iterationNo += 1
        ##print('iterationNo = ', iterationNo)
    return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    intensity = cv.imread('../images/flower.png',0)
    cv.imshow("flowers", intensity)
    data = intensity.flatten()
    data.shape = (data.shape[0],1)
    for k in [2,4,6]:
        index, centers = myKmeans(data, k)
        index.shape = intensity.shape
        segmentedImage = index.astype(np.uint8)

        for cInx in range(k):
            ##setting color of segment as the centroids color
            segmentedImage[np.where(index == cInx)] = centers[cInx]

        cv.imshow("T3-a: segmented image k = %d" % k,segmentedImage)



def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    cv.imshow("original image", img)


    flat_img = img.flatten()
    ##making the data matrix
    flat_img.shape = ( int(flat_img.shape[0]/3), 3)

    for k in [2,4,6]:
        index, centers = myKmeans(flat_img, k)
        index.shape = (img.shape[0], img.shape[1])

        segmentedImage = np.zeros(img.shape,dtype=np.uint8)

        for cInx in range(k):
            ##setting color of segment as the centroids color
            segmentedImage[np.where(index==cInx)] = centers[cInx]

        cv.imshow("T3-b: segmented image k = %d" % k, segmentedImage)


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    cv.imshow("original image", img)
    k = 6

    ####adding location data to img

    ##calculating normalization factor
    max_side = max(img.shape[0],img.shape[1])
    factor = 255.0 / max_side
    #gen coordinate matrix
    coor_mat_l, coor_mat_c =  np.fromfunction(lambda i, j: (i*factor, j*factor), (img.shape[0],img.shape[1]), dtype=np.float32)

    ##creates two matrices: mat_l containing the line coordinates, mat_c the column ones
    coor_mat_l = coor_mat_l.astype(np.uint8)
    coor_mat_l = coor_mat_l.flatten()
    coor_mat_l.shape = (coor_mat_l.shape[0],1)
    coor_mat_c = coor_mat_c.astype(np.uint8)
    coor_mat_c = coor_mat_c.flatten()
    coor_mat_c.shape = (coor_mat_c.shape[0], 1)


    flat_img = img.flatten()
    ##making the data matrix
    flat_img.shape = ( int(flat_img.shape[0]/3), 3)

    for k in [2, 4, 6]:
        imgWithCoord = np.concatenate((flat_img,coor_mat_l, coor_mat_c), axis=1)
        index, centers = myKmeans(imgWithCoord, k)
        index.shape = (img.shape[0], img.shape[1])

        segmentedImage = np.zeros(img.shape,dtype=np.uint8)

        for cInx in range(k):
            ##setting color of segment as the centroids color
            segmentedImage[np.where(index==cInx)] = centers[cInx][0:3]
        cv.imshow("T3-c segmented image, cluster color, k = %d"%k, segmentedImage)
        segmentedImageIndices = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for cInx in range(k):
            ##setting color of segment as the centroids color
            segmentedImageIndices[np.where(index==cInx)] = cInx*(255/k)
        cv.imshow("T3-c segmented image, cluster index, k = %d"% k, segmentedImageIndices)



##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")

    W = np.zeros((8,8), dtype=np.float32)
    W[1,0] = 1
    W[2,0] = 0.2
    W[3,0] = 1
    W[2,1] = 0.1
    W[3,2] = 1
    W[4,1] = 1
    W[6,2] = 0.3
    W[5,2] = 1
    W[5,3] = 1
    W[6,5] = 1
    W[7,6] = 1
    W[6,4] = 1
    W[7,4] = 1
    W = np.maximum(W, W.transpose()) ## copy values to other side of matrix
    #print("W:",W)
    D = np.diag(W.sum(axis= 1))

    ## D is diagonal so its square root is the element-wise square root
    Dsq = np.sqrt(D)
    Dsq_inv = np.reciprocal(Dsq, where=np.diag([True] * D.shape[0]))


    mat = np.matmul( np.matmul(Dsq_inv, (D-W)),Dsq_inv)
    retval,evalues,evectors = cv.eigen(mat)

    #print(evalues)
    #print(evectors)

    ##eigenvalues are stored in descending order. eigenvectors in the same order as correspoding eigenvalues
    vector = evectors[-2] ##get evector corresponding to second smallest eigenvalue
    ##obatining y
    vector.reshape(8,1)
    y = np.matmul(Dsq_inv,vector)
    #print(vector)
    print("vector y is:")
    print(y)

    print("Calculated cut:")
    cut = np.zeros((8,1))
    inxC1 = np.where(y < 0)[0]
    inxC2 = np.where(y > 0)[0]

    print("cluster 1:", inxC1+1)
    print("cluster 2:", inxC2 + 1)

    cutcost = 0.0
    for i1 in inxC1:
        for i2 in inxC2:
            cutcost += W[i1,i2]

    volC1 = 0.0
    for i1 in inxC1:
        volC1 += D[i1,i1]
    volC2 = 0.0
    for i2 in inxC2:
        volC2 += D[i1, i1]
    normcut = cutcost*(1/volC1 + 1/volC2)
    print("normalized cut cost:", normcut)


##############################################
##############################################
##############################################


task_1_a()
task_1_b()
task_2()
task_3_a()
task_3_b()
task_3_c()
task_4_a()

cv.waitKey(0)

cv.destroyAllWindows()
