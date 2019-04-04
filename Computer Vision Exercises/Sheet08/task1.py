import numpy as np
import utils

'''
def normalize_size(mean,desiredavgdist):
    nlandmarks = mean.shape[0]
    x_vals = mean[:,0]
    y_vals = mean[:,1]

    #recenter hand
    #x_vals -= 0.5#np.mean(x_vals)
    #y_vals -= 0.5#np.mean(y_vals)
    print("mean xy", np.mean(x_vals), np.mean(y_vals))

    ##get distance of points from origin
    dist = np.sqrt(np.power(x_vals,2) + np.power(y_vals, 2))

    avgdist = np.sum(dist)/nlandmarks
    print("avgdist:", avgdist)
    #fac = desiredavgdist/avgdist
    fac = 1/avgdist


    return mean*fac

def get_angle_landmark(hand, landmarkix):
    return np.arctan2(hand[landmarkix,1],hand[landmarkix,0])

def normalize_mean(mean_,desiredavgdist, anglefirst):
    mean = mean_.copy()
    mean.shape = (1,mean_.shape[0], mean_.shape[1])
    mean = normalize_hands_center_mass(mean)
    mean.shape = (mean_.shape[0], mean_.shape[1])
    ##account for rotation
    angle = get_angle_landmark(mean, 0)

    print("current angle:", angle)
    print("original angle", anglefirst)


    rotateby = anglefirst - angle
    print("rotate by:",rotateby )
    xold = mean[0][0]
    yold =  mean[0][1]

    rotmatrix = np.zeros((2,2))
    rotmatrix[0,0] = np.cos(rotateby)
    rotmatrix[0,1] =  - np.sin(rotateby)

    rotmatrix[1,0] = np.sin(rotateby)
    rotmatrix[1,1] = np.cos(rotateby)
    for coor in range(mean.shape[0]):
        mean[coor] = np.matmul(rotmatrix , mean[coor])

    return mean




def normalize_hands_center_mass(kpts):

    ##normalize size
    kpts = normalize_size(kpts, 1)

    ##shift centers of mass to irigin
    summ = np.sum(kpts, axis=1) ##sum over x and y- shape: (#hands, 2)
    avg = summ/kpts.shape[1]
    for hnd in range(kpts.shape[0]):
        kpts[hnd] -= avg[hnd]



    return kpts
'''

def normalize_mean(mean):
    nlandmarks = mean.shape[0]
    x_vals = mean[:,0]
    y_vals = mean[:,1]

    #recenter hand
    x_vals -= 0.5#np.mean(x_vals)
    y_vals -= 0.5#np.mean(y_vals)

    ##get distance of points from origin
    dist = np.sqrt(np.power(x_vals,2) + np.power(y_vals, 2))

    avgdist = np.sum(dist)/nlandmarks

    #fac = desiredavgdist/avgdist
    fac = 1/avgdist
    return mean*fac

# ========================== Mean =============================
def calculate_mean_shape(kpts_):

    #return kpts[0]
    return np.average(kpts_, axis=0)

def avgdist_hands(kpts):
    dist = 0.0
    for k in range(kpts.shape[0]):
         dist += calculate_avg_distance(kpts[k])
    return dist / kpts.shape[0]


def calculate_avg_distance(hand):
     x_vals = hand[:,0]
     y_vals = hand[:,1]

     ##x_vals -= np.mean(x_vals)
     ##y_vals -= np.mean(y_vals)
     nlandmarks = hand.shape[0]
     dist = np.sqrt(np.power(x_vals,2) + np.power(y_vals, 2))

     return np.sum(dist)/nlandmarks


# ====================== Main Step ===========================
def procrustres_analysis_step(kpts_, reference_mean):
    # Happy Coding
    kpts = kpts_.copy()
    num_dim = kpts.shape[2]
    num_instances = kpts.shape[0]
    num_landmarks = kpts.shape[1]

    error_sum= 0.0

    ##create b from mean coordinates
    B = np.zeros((num_landmarks*num_dim,1))
    B[0:num_landmarks,0] = reference_mean[:,0]
    B[num_landmarks:,0] = reference_mean[:,1]

    for i in range(num_instances):
        #create A from instance data
        A = np.zeros((num_landmarks*num_dim,6))
        ##filling for x' (dimension 1)
        A[0:num_landmarks,0] = kpts[i,:, 0]
        A[0:num_landmarks,1] = kpts[i,:, 1]
        A[0:num_landmarks,4] = 1

        ##filling for y' (dimension 2)
        A[num_landmarks:,2] = kpts[i,:, 0]
        A[num_landmarks:,3] = kpts[i,:, 1]
        A[num_landmarks:,5] = 1



        ## least squares problem with Ax=b
        x, errorsq, rank, s = np.linalg.lstsq(A,B)
        #print(errorsq)
        new_pos = np.matmul(A,x)
        kpts[i,:,0] = new_pos[0:num_landmarks,0]
        kpts[i,:,1] = new_pos[num_landmarks:,0]
        # error_sum += errorsq[0]

    return kpts,error_sum
    ##apply calculated x to obtain updated positions



# =========================== Error ====================================

def compute_avg_error(kpts, mean_shape):
    dist = kpts-mean_shape
    dist = dist * dist
    err = np.sum(dist, axis=2)
    avg_err = np.array(np.average(err))
    return avg_err



# ============================ Procrustres ===============================

def procrustres_analysis(kpts, max_iter=int(1e3), min_error=1e-5):

    #kpts = normalize_hands_center_mass(kpts)

    num_instances = kpts.shape[0]

    aligned_kpts = kpts.copy()
    mean = calculate_mean_shape(aligned_kpts)
    continue_iteration = True
    all_means = mean.copy()
    all_means.shape = (1,56,2)

    #original_angle = get_angle_landmark(kpts[0], 0)
    reference_mean = kpts[0].copy()

    iter = 0
    while iter < max_iter and continue_iteration:



        reference_mean = normalize_mean(reference_mean)#,1, original_angle)

        threed_mean = reference_mean.copy()
        threed_mean.shape =(1, 56, 2)

        all_means = np.append(all_means, threed_mean,axis=0)
        # align shapes to mean shape
        #aligned_kpts = procrustres_analysis_step(aligned_kpts, reference_mean)

        aligned_kpts,error_sum = procrustres_analysis_step(kpts, reference_mean)

        #
        reference_mean = calculate_mean_shape(aligned_kpts)

        RMS = compute_avg_error(aligned_kpts, reference_mean)

        ##################### Your Part Here #####################
        if(RMS < min_error):
            continue_iteration = False

        iter+=1
        ##########################################################
    #mean = calculate_mean_shape(aligned_kpts)
    #print("shap aligned",aligned_kpts.shape)
    #mean.shape=  (1, mean.shape[0], mean.shape[1])
    #print(mean)
    #print("all means shape:",all_means.shape)

    #utils.visualize_hands(mean,"first mean",delay=0.1)
    #utils.visualize_hands(all_means,"means",delay=0.00001)
    #utils.visualize_hands(kpts,"input",delay=0.001)
    # visualize

    utils.visualize_hands(aligned_kpts,"aligned", delay=0.0001)
    # visualize mean shape
    last_mean = all_means[-1]
    last_mean.shape = (1, mean.shape[0], mean.shape[1])
    utils.visualize_hands(last_mean,"mean",delay=0.00001)
    a = input("press any key to close windows")

    return aligned_kpts
