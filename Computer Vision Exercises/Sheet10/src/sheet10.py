
import numpy as np
import os
import cv2 as cv


MAX_ITERATIONS = 1000 # maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
EPSILON = 0.002 # the stopping criterion for the difference when performing the Horn-Schuck algorithm
EIGEN_THRESHOLD = 0.01 # use as threshold for determining if the optical flow is valid when performing Lucas-Kanade

def load_FLO_file(filename):

    assert os.path.isfile(filename), 'file does not exist: ' + filename
    flo_file = open(filename,'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    assert magic == 202021.25,  'Magic number incorrect. .flo file is invalid'
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    #the float values for u and v are interleaved in row order, i.e., u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...,
    # in total, there are 2*w*h flow values
    data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    flo_file.close()
    return flow




#***********************************************************************************
#implement Lucas-Kanade Optical Flow
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# window_size: the number of points taken in the neighborhood of each pixel
# returns the Optical flow based on the Lucas-Kanade algorithm
def Lucas_Kanade_flow(frames, Ix, Iy, It, window_size):

    IxIx = np.multiply(Ix, Ix)
    IxIy = np.multiply(Ix , Iy)
    IyIy = np.multiply(Iy , Iy)
    IxIt = np.multiply(Ix , It)
    IyIt = np.multiply(Iy ,It)

    size_y, size_x= frames[0].shape



    optFlow = np.zeros((size_y,size_x,2))

    asum = np.zeros((size_y,size_x))
    print("Computing, Please wait...")
    for y in range(size_y):
        for x in range(size_x):
            ##calculating upper-left limit
            min_x = max(0,(x-int(window_size/2)) )
            min_y = max(0,(y-int(window_size/2)) )
            ##calculating lower right limit
            max_x = min(size_x,(x+int(window_size/2)+1) )
            max_y = min(size_y,(y+int(window_size/2)+1) )
            ##calculate displacement at x,y
            sumIxIx = np.sum(IxIx[min_y:max_y,min_x:max_x])
            sumIxIy = np.sum(IxIy[min_y:max_y,min_x:max_x])
            sumIyIy = np.sum(IyIy[min_y:max_y,min_x:max_x])
            sumIxIt = np.sum(IxIt[min_y:max_y,min_x:max_x])
            sumIyIt = np.sum(IyIt[min_y:max_y,min_x:max_x])


            A = np.array([sumIxIx,sumIxIy,sumIxIy,sumIyIy])
            A.shape = (2,2)
            ##calculates eigenvalues
            evalues = np.linalg.eigvalsh(A)
            if(evalues[0]> EIGEN_THRESHOLD and evalues[1]> EIGEN_THRESHOLD ):
                B = np.array([-sumIxIt, -sumIyIt])
                B.shape = (2,1)
                displacement, errorsq, rank, s = np.linalg.lstsq(A,B)

                optFlow[y,x,0] = displacement[0,0]
                optFlow[y,x,1] = displacement[1,0]
            else:
                optFlow[y,x,0] = 0
                optFlow[y,x,1] = 0
    return optFlow


#***********************************************************************************
#implement Horn-Schunck Optical Flow
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# window_size: the number of points taken in the neighborhood of each pixel
# returns the Optical flow based on the Horn-Schunck algorithm
def Horn_Schunck_flow(Ix, Iy, It):
    i = 0
    diff = 1
    size_y, size_x= Ix.shape

    u = np.zeros((size_y,size_x),dtype=np.float64)
    v = np.zeros((size_y,size_x),dtype=np.float64)
    IxIx = Ix*Ix
    IyIy = Iy*Iy

    while i<MAX_ITERATIONS and diff > EPSILON: #Iterate until the max number of iterations is reached or the difference is less than epsilon
        i += 1

        u_bar = u + get_laplacian(u)
        v_bar = v + get_laplacian(v)

        divisor  = (1+IxIx+IyIy)
        term = (Ix*u_bar + Iy*v_bar + It)
        new_u = u_bar - Ix*(term/divisor)
        new_v = v_bar - Iy*(term /divisor)


        ##calculating error
        diff = np.sum(np.abs(u - new_u) + np.abs(v - new_v))
        u = new_u
        v = new_v
        print("diff:", diff)


    optFlow = np.stack((u,v),axis=2)
    return optFlow



def get_laplacian(matr):
    kernel = np.array([0,0.25, 0, 0.25,-1,0.25,0,0.25,0])
    kernel.shape = (3,3)
    return cv.filter2D(matr, -1, kernel)


#calculate the angular error here
def calculate_angular_error(estimated_flow, groundtruth_flow):
    anglesEst = np.degrees(np.arctan2(estimated_flow[:,:,1],estimated_flow[:,:,0]))
    anglesGT = np.degrees(np.arctan2(groundtruth_flow[:,:,1],groundtruth_flow[:,:,0]))
    err = np.abs(anglesGT - anglesEst).mean()
    return err


def show_HSV_space():
    y = 300
    x = 300
    outHSV = np.zeros((y,x,3),dtype=np.uint8)
    outHSV[:,:,1] = 200 #saturation
    outHSV[:,:,2] = 200
    angles = np.zeros((y,x), dtype=np.float32)
    for yinx in range(y):
        for xinx in range(x):
            angles[yinx,xinx] = np.arctan2(150-yinx,xinx-150)
    degrees = np.degrees(angles)
    outHSV[:,:,0] = degrees.astype(np.uint8)
    color = cv.cvtColor(outHSV, cv.COLOR_HSV2BGR)
    return color
#function for converting flow map to to BGR image for visualisation
def flow_map_to_bgr(flo):
    y,x,z = flo.shape

    outHSV = np.zeros((y,x,3),dtype=np.uint8)
    outHSV[:,:,1] = 200 #saturation
    outHSV[:,:,2] = 200
    ##optflow is (u,v)
    angles = np.degrees(np.arctan2(flo[:,:,1],flo[:,:,0]))/2.0
    lengths =np.sqrt(flo[:,:,0]*flo[:,:,0] + flo[:,:,1]*flo[:,:,1])
    max_length = lengths.max()
    print("max length is ", max_length)
    outHSV[:,:,2] = (255*(lengths/max_length)).astype(np.uint8)

    outHSV[:,:,0] = angles.astype(np.uint8)
    color = cv.cvtColor(outHSV, cv.COLOR_HSV2BGR)
    return color



if __name__ == "__main__":
    # read your data here and then call the different algorithms, then visualise your results
    WINDOW_SIZE = [15, 15]  #the number of points taken in the neighborhood of each pixel when applying Lucas-Kanade
    gt_flow = load_FLO_file('../data/groundTruthOF.flo')
    img1 = cv.imread("../data/frame1.png",cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../data/frame2.png",cv.IMREAD_GRAYSCALE)

    cv.imshow("img1", img1)
    cv.imshow("img2", img2)


    ##derivatives are computed over all axis
    ##based on http://image.diku.dk/imagecanon/material/HornSchunckOptical_Flow.pdf
    kX = np.array([[-0.25, 0.25],[-0.25,0.25]])
    kY = np.array([[-0.25, -0.25],[0.25,0.25]])
    kT = np.array([[0.25,0.25],[0.25,0.25]])

    I_x = cv.filter2D(img1, cv.CV_64F, kX) + cv.filter2D(img2, cv.CV_64F, kX)
    I_y = cv.filter2D(img1, cv.CV_64F, kY) + cv.filter2D(img2, cv.CV_64F, kY)
    ##calculating It
    temp_diff = img2-img1
    I_t = cv.filter2D(temp_diff, cv.CV_64F, kT)
    #I_t = img2.astype(np.float32) -img1.astype(np.float32)
    print(I_x.min())
    print(I_t)
    print(type(I_t))
    frames = np.stack((img1,img2))

    LKFlow = Lucas_Kanade_flow(frames, I_x, I_y, I_t, WINDOW_SIZE[0])

    cv.imshow("LKFlow", flow_map_to_bgr(LKFlow))
    print("angular error LK:",calculate_angular_error(LKFlow, gt_flow))

    HSFlow = Horn_Schunck_flow(I_x, I_y, I_t)
    cv.imshow("HSFlow", flow_map_to_bgr(HSFlow))
    print("angular error HS:",calculate_angular_error(HSFlow, gt_flow))

    cv.imshow("flow ground truth", flow_map_to_bgr(gt_flow))



    cv.waitKey(0)
