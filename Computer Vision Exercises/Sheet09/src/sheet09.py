import cv2
import numpy as np
import random

#   =======================================================
#                   Task1
#   =======================================================
def task1():

    img1Color = cv2.imread('../images/building.jpeg')
    cv2.imshow("image", img1Color)
    img1 =  cv2.cvtColor(img1Color,cv2.COLOR_BGR2GRAY)

    #apply gaussian filter
    img1 =  cv2.GaussianBlur(img1, (3, 3), 0)

    # compute structural tensor
    I_x = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=3)
    I_y = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=3)
    cv2.imshow("sobelx", I_x)
    I_xy = I_x * I_y
    I_xx = I_x * I_x
    I_yy = I_y * I_y
    print()
    normI = np.log(np.abs(I_xy))
    normI = (normI / np.max(normI))*255



    cv2.imshow("Ixy log",normI.astype(np.uint8))


    box = np.ones((3,3))
    box /= 9.0
    C_xx = cv2.filter2D(I_xx,-1,box)
    C_xy = cv2.filter2D(I_xy,-1,box)
    C_yy = cv2.filter2D(I_yy,-1,box)

    assert(img1.shape == C_xx.shape)
    M = lambda posY,posX : np.array([[C_xx[posY][posX], C_xy[posY][posX]],[C_xy[posY][posX], C_yy[posY][posX]]])
    k = 0.09
    F = lambda M: max(np.abs(np.linalg.det(M)) - k*(np.trace(M)**2),0)
    print(img1.shape)


    #Harris Corner Detection

    fitness = np.zeros(img1.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            fitness[i,j] = F(M(i,j))
    ##visualization response function
    respFuncVis = np.log(fitness)
    respFuncVis = (respFuncVis / np.max(respFuncVis))*255
    cv2.imshow("response function (log)", respFuncVis.astype(np.uint8))

    ##thresholding
    threshold = 6000000
    filtered = fitness.copy()
    filtered[np.where(filtered < threshold)] = 0

    fit = np.log(filtered)

    fit = (fit / np.max(fit))*255
    cv2.imshow("response function thresholded(log)", fit.astype(np.uint8))


    ###filtering. keeping only local maxima
    for i in range(img1.shape[0]):
        i_lw_b = max(0,i-1)
        i_up_b = min(img1.shape[0], i+2)
        for j in range(img1.shape[1]):
            j_lw_b = max(0, j - 1)
            j_up_b = min(img1.shape[1], j + 2)

            mx = np.max(filtered[i_lw_b:i_up_b,j_lw_b:j_up_b])
            if(filtered[i,j] != mx):
                filtered[i,j] = 0

    maxima = filtered.copy()
    maxima = np.log(maxima)
    maxima = (maxima / np.max(maxima))*255
    cv2.imshow("corners (log)", maxima.astype(np.uint8))

    corners_H = img1Color.copy()
    corners_H[np.where(filtered>0)] = np.array([0,0,255])
    cv2.imshow("Harris corners", corners_H)


    #Forstner Corner Detection
    w_min = 500.0
    q_min = 0.85


    ##small noise to prevent div by zero
    epsilon =0.0001
    W = np.zeros(img1.shape)
    Q = np.zeros(img1.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            m = M(i,j)
            detM = np.linalg.det(m)
            Tm = np.trace(m) + epsilon
            assert(Tm != 0 )
            W[i,j] = detM/Tm
            assert( not np.isnan(W[i,j]))
            Q[i,j] = (4*detM)/ Tm**2



    ##thresholding
    W[np.where(W<w_min)] = 0
    Q[np.where(Q<q_min)] = 0


    wshow = W.copy()
    #wshow = np.log(wshow)
    wshow = (wshow / np.max(wshow))*255
    cv2.imshow("W", wshow.astype(np.uint8))
    qshow = Q.copy()

    #qshow = np.log(qshow)
    qshow = (qshow / np.max(qshow))*255
    cv2.imshow("Q", qshow.astype(np.uint8))

    corners_F_mask = Q*W
    corners_F = img1Color.copy()

    corners_F[np.where(corners_F_mask>0)] = np.array([0,0,255])
    cv2.imshow("Foerstner corners ", corners_F)
    print("Press Any Key to Continue")
    cv2.waitKey(0)
    print("Press Wait... Calculating.")



#   =======================================================
#                   Task2
#   =======================================================
def task2():
### since we could not find a reasonable documentation of this module we
##used this as main reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    img1 = cv2.imread('../images/mountain1.png')
    img2 = cv2.imread('../images/mountain2.png')

    im1gr= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    im2gr= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #extract sift keypoints and descriptors

    s = cv2.xfeatures2d.SIFT_create()
    img1Key = s.detect(im1gr)
    img1Key, img1Des = s.compute(im1gr, img1Key)
    img2Key = s.detect(im2gr)
    img2Key, img2Des = s.compute(im2gr, img2Key)

    ##descriptor is a tuple vector matrix one line for each keypoint

    ## in case we want to plot the keypoints
    #imk=cv2.drawKeypoints(im1gr,img1K,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ##cv2.imshow("my kp", imk)

    # own implementation of matching
    #we define img1 as query image
    ## img2 as train image


    # create BFMatcher object
    bf = cv2.BFMatcher()


    dmatchK1 = [[None, None] for x in range(img1Des.shape[0])] ## kptindex: [best,scnd]
    dmatchK2 = [[None, None] for x in range(img2Des.shape[0])]
    # Match descriptors.
    for kp1 in range(img1Des.shape[0] ):
        #print(f"calculating matches : {kp1/img1Des.shape[0]} pct")
        distances =  np.sum(np.power(img1Des[kp1]- img2Des, 2),axis = 1)
        #        first = None
        #second = None
        for kp2, dist in enumerate(distances):
            if(dmatchK1[kp1][0] == None):
                dmatchK1[kp1][0] = cv2.DMatch(kp1,kp2, dist)
            elif(dmatchK1[kp1][1] == None):
                dmatchK1[kp1][1] = cv2.DMatch(kp1,kp2, dist)
            elif dist < dmatchK1[kp1][0].distance:
                dmatchK1[kp1][1] = dmatchK1[kp1][0]
                dmatchK1[kp1][0] = cv2.DMatch(kp1,kp2, dist)
            elif dist > dmatchK1[kp1][0].distance and dist < dmatchK1[kp1][1].distance:
                dmatchK1[kp1][1] = cv2.DMatch(kp1,kp2, dist)

            if(dmatchK2[kp2][0] == None):
                dmatchK2[kp2][0] = cv2.DMatch(kp1,kp2, dist)
            elif(dmatchK2[kp2][1] == None):
                dmatchK2[kp2][1] = cv2.DMatch(kp1,kp2, dist)
            elif dist < dmatchK2[kp2][0].distance:
                dmatchK2[kp2][1] = dmatchK2[kp2][0]
                dmatchK2[kp2][0] = cv2.DMatch(kp1,kp2, dist)
            elif dist > dmatchK2[kp2][0].distance and dist < dmatchK2[kp2][1].distance:
                dmatchK2[kp2][1] = cv2.DMatch(kp1,kp2, dist)



    symmetric_matches = []

    for kp1, match in enumerate(dmatchK1):
        best_other = match[0].trainIdx
        scnbest_other = match[1].trainIdx
        if(dmatchK2[best_other][0].queryIdx == kp1):
            symmetric_matches.append(match)

    good_matches = []

    for first,second in symmetric_matches:
        if(first.distance < 0.4*second.distance):
            good_matches.append([first])


    good_matches = sorted(good_matches, key= lambda x : x[0].distance)

    img3 = cv2.drawMatchesKnn(img1, img1Key, img2, img2Key, good_matches,None,flags=2)

    # display matched keypoints
    cv2.imshow("keypoint matches", img3)


    #  =======================================================
    #                          Task-3
    #  =======================================================

    nSamples = 4;
    nIterations = 20;
    thresh = 0.1;
    minSamples = 4;
    height, width, _ = img2.shape

    inliers = 0
    keep_h = 0

    #  /// RANSAC loop
    good_matches = np.array(good_matches).flatten()
    for i in range(nIterations):

        print('iteration '+str(i))

        #randomly select 4 pairs of keypoints
        nump = 4
        chosen = np.random.choice(good_matches, nump)


        points1 = np.zeros((nump,2), np.float32)
        points2 = np.zeros((nump,2), np.float32)
        for j in range(0,nump):
            points1[j] = img1Key[chosen[j].queryIdx].pt
            points2[j] = img2Key[chosen[j].trainIdx].pt

        #compute transformation and warp img2 using it
        h = cv2.getPerspectiveTransform(points2, points1)

        timg1 = np.copy(img1)

        nInliers = 0
        for j in range(0,good_matches.shape[0]):
            kp1 = img1Key[good_matches[j].queryIdx].pt
            kp2 = img2Key[good_matches[j].trainIdx].pt

            kp2 = np.array([np.array([[kp2[0],kp2[1]]], dtype="float32")])
            kp2 = cv2.perspectiveTransform(kp2, h).flatten()

            dist = np.sqrt((kp1[0]-kp2[0])**2 + (kp1[1] - kp2[1])**2)
            if(dist <= thresh):
                nInliers += 1

            cv2.circle(timg1,(int(kp2[0]), int(kp2[1])), 3, (255,0,0), -1)
            cv2.circle(timg1,(int(kp1[0]), int(kp1[1])), 3, (0,0,255), -1)

        cv2.imshow("Keypoints from Image 1 and Image 2 on same Image ", timg1)

        #count inliers and keep transformation if it is better than the best so far
        if nInliers > inliers:
            inliers = nInliers
            keep_h = h
        else:
            h = keep_h

        '''
        print("New Transform:", h)
        print("Keep Transform:", keep_h)
        '''
        warped = cv2.warpPerspective(img2, h, (width, height))

        cmbImg = warped + img1
        cv2.imshow("warped image", warped)
        cv2.imshow("comb image", cmbImg)
        #        cv2.wait(100)
        cv2.waitKey(100)


    #apply best transformation to transform img2
    warped = cv2.warpPerspective(img2, keep_h, (width, height))
    cmbImg = warped
    inds = np.where(cmbImg == 0)
    cmbImg[inds] = img1[inds]

    #display stitched images
    cv2.imshow("Final Image", cmbImg)
    cv2.waitKey(0)

    exit()

task1()
task2() #also task 3
