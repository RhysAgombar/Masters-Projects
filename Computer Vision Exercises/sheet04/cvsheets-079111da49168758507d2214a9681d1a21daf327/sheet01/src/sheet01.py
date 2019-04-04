import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import sys


if __name__ == '__main__':
    img_path = sys.argv[1]


#    =========================================================================
#    ==================== Task 1 =================================
#    =========================================================================
    print('------------------');
    print('Task 1:');
    print('------------------');
    # a
    print("------")
    print("a")
    print("------")
    img = cv.cvtColor(cv.imread(img_path), cv.COLOR_RGB2GRAY)

    def integ(img):
        img_int = np.zeros(img.shape, dtype=int)

        img_int[0,0] = img[0,0]

        for i in range(1, img_int.shape[0]):
            img_int[i,0] = img[i,0] + img_int[i-1,0]

        for j in range(1, img_int.shape[1]):
            img_int[0,j] = img[0,j] + img_int[0,j-1]

        for i in range(1, img_int.shape[0]):
            for j in range(1, img_int.shape[1]):
                img_int[i,j] = img[i,j] + img_int[i,j-1] - img_int[i-1,j-1] + img_int[i-1,j]

        return img_int

    img_int = integ(img)

    print("Image values for first 5x5")
    print(img[0:5,0:5])
    print("Integral values for first 5x5")
    print(img_int[0:5,0:5])
    cv.imshow("Task 1: Integral Image", img_int)

    #b
    #i
    print("------")
    print("b")
    print("------")
    print("Mean Grey Value by Sum: ", np.sum(img)/(img.shape[0] * img.shape[1]))

    #ii
    pos = cv.integral(img).shape
    print("Mean Grey Value by built-in Integral: ", cv.integral(img)[pos[0]-1, pos[1]-1]/(img.shape[0] * img.shape[1]))

    #iii
    img_int = integ(img)
    pos = img_int.shape
    print("Mean Grey Value by Our Integral Function: ", img_int[pos[0]-1, pos[1]-1]/(img_int.shape[0] * img_int.shape[1]))


    #c
    print("------")
    print("c")
    print("------")
    squares = []
    num_squares = 10
    for i in range(0,num_squares):
        ypos = random.randint(0, img.shape[0] - 100)
        xpos = random.randint(0, img.shape[1] - 100)
        squares.append(img[ypos:ypos+100, xpos:xpos+100])
        #cv.imshow("test " + str(i), positions[i])

    print("Mean Grey by Sum, Random Squares, 10 Iterations")
    start = time.clock()
    for i in range(0,num_squares):
        sum_mean = np.sum(squares[i])/(squares[i].shape[0] * squares[i].shape[1])
        print(sum_mean)
    end = time.clock()
    print("Elapsed Time: ", end-start)

    print("Mean Grey by built-in Integral, Random Squares, 10 Iterations")
    start = time.clock()
    for i in range(0,num_squares):
        pos = squares[i].shape
        sum_mean = cv.integral(squares[i])[pos[0]-1, pos[1]-1]/(squares[i].shape[0] * squares[i].shape[1])
        print(sum_mean)
    end = time.clock()
    print("Elapsed Time: ", end-start)

    print("Mean Grey by Our Integral Function, Random Squares, 10 Iterations")
    start = time.clock()
    for i in range(0,num_squares):
        img_int = integ(squares[i])
        pos = img_int.shape
        sum_mean = img_int[pos[0]-1, pos[1]-1]/(img_int.shape[0] * img_int.shape[1])
        print(sum_mean)
    end = time.clock()
    print("Elapsed Time: ", end-start)

#    =========================================================================
#    ==================== Task 2 =================================
#    =========================================================================
    print('Task 2:');

    img2 = cv.imread(img_path)
    intensity = cv.cvtColor(img2, cv.COLOR_BGR2HSV)[:,:,2]

    eq_builtin = cv.equalizeHist(intensity)
    cv.imshow("built in eq.",eq_builtin)

    count, binnum = np.histogram(intensity,bins=256)
    cdf = count.cumsum()
    cdf_norm = cdf / float(cdf[-1])
    eq_intensities =(cdf_norm * 255).astype(np.uint8)
    f = np.vectorize(lambda v: eq_intensities[v])
    eq_impl = f(intensity)
    cv.imshow("implemented eq.",eq_impl)




#    =========================================================================
#    ==================== Task 4 =================================
#    =========================================================================
    print('Task 4:');
    img4 = cv.imread(img_path)
    intensity = cv.cvtColor(img4, cv.COLOR_BGR2HSV)[:,:,2]
    cv.imshow("T4: original",intensity)

    sigma =  2.0*np.sqrt(2)
    filt_gaussian_a = cv.GaussianBlur(intensity, (3,3), sigma )
    cv.imshow("T4: gaussianBlur(a)",filt_gaussian_a)



    ##blurring with filter2d
    
    #calculating kernel values
    gaussian = lambda x,y:  (1.0 / 2*np.pi*(sigma**2))*np.exp(- (x**2 + y**2)/(2*sigma**2))
    kernel = np.zeros((3,3),dtype=float)
    for i in range(3):
        for j in range(3):

            kernel[i][j] = gaussian((i-1), (j-1))
    #normalizing kernel (sum to 1)
    kernel = kernel/kernel.sum()
    filt_gaussian_b = cv.filter2D(intensity, -1, kernel)
    cv.imshow("T4: filter2D(b)",filt_gaussian_b)

    ##bluring with sepfilter2D
    gaussian1d = lambda x:  (1.0 / np.sqrt(2*np.pi*(sigma**2)))*np.exp(- x**2/(2*sigma**2))
    kernel = np.zeros(3,dtype=float)
    for i in range(3):
        kernel[i]= gaussian1d(i-1)
    kernel = kernel/kernel.sum()
    filt_gaussian_c = cv.sepFilter2D(intensity, -1, kernel, kernel)
    print(filt_gaussian_c)
    cv.imshow("T4: sepFilter2D(c)",filt_gaussian_c)

    ##calculating differences
    delta_a_b = np.abs(filt_gaussian_a - filt_gaussian_b)
    delta_a_c = np.abs(filt_gaussian_a - filt_gaussian_c)
    delta_b_c = np.abs(filt_gaussian_b - filt_gaussian_c)

    text = "maximum pixel error for pair %s is %d"
    print(text % ("(a,b)", delta_a_b.max()))
    print(text % ("(a,c)", delta_a_c.max()))
    print(text % ("(b,c)", delta_b_c.max()))

#    cv.imshow("diff_a_b", delta_a_b)
#    cv.imshow("diff_a_c", delta_a_c)
#    cv.imshow("diff_b_c", delta_b_c)


#    =========================================================================
#    ==================== Task 5 =================================
#    =========================================================================
    print('------------------');
    print('Task 5:');
    print('------------------');
    mid = 2
    dim = 5
    sigma = 2
    g2 = np.zeros((dim,dim))
    for i in range(0,dim):
        for j in range(0,dim):
            g2[i,j] = (1/(2*np.pi*sigma**2))*np.exp(-1*((i-mid)**2 + (j-mid)**2)/(2*sigma**2))

    print("g2 kernel:")
    print(g2)
    #plt.imshow(g2)
    #plt.title("Gaussian, Sigma = 2")
    #plt.show()

    sigma = (2*np.sqrt(2))
    g2sq = np.zeros((dim,dim))
    for i in range(0,dim):
        for j in range(0,dim):
            g2sq[i,j] = (1/(2*np.pi*sigma**2))*np.exp(-1*((i-mid)**2 + (j-mid)**2)/(2*sigma**2))

    print("g2sqrt kernel:")
    print(g2sq)
    #plt.imshow(g2sq)
    #plt.title("Gaussian, Sigma = 2*sqrt(2)")
    #plt.show()

    #--------------
    #a
    ia = cv.filter2D(cv.filter2D(img,-1,g2), -1, g2)
    cv.imshow("Task 5: Filter twice with sigma = 2", ia)
    #b
    ib = cv.filter2D(img, -1, g2sq)
    cv.imshow("Task 5: Filter once with sigma = 2*sqrt(2)", ib)

    pixeldist = abs(ia-ib)
    print("Maximum Pixel Error: ", np.amax(pixeldist))




#    =========================================================================
#    ==================== Task 7 =================================
#    =========================================================================
    print('Task 7:');

    print('------------------');
    print('Task 7:');
    print('------------------');

    nImg = np.zeros(img.shape, dtype=np.uint8)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            val = random.random()
            if (val < 0.30):
                val = random.random()
                if (val < 0.50):
                    nImg[i,j] = 0
                else:
                    nImg[i,j] = 255
            else:
                nImg[i,j] = img[i,j]

    cv.imshow("Task 7: 30% Salt and Pepper Noise", nImg)
    cv.imshow("Task 7: Original Image", img)
    dim = 5 #this was chosen as the filter size. It minimizes the mean grey value distance to the original image.

    g = cv.GaussianBlur(nImg, (dim,dim), 0)
    cv.imshow("Task 7: Gaussian Filter, Sigma = 2, Size = " + str(dim), g)
    m = cv.medianBlur(nImg, dim)
    cv.imshow("Task 7: Median Blur Filter, Size = " + str(dim), m)
    b = cv.bilateralFilter(nImg, dim, 150, 150)
    cv.imshow("Task 7: Bilateral Filter, Size = " + str(dim), b)


    pos = cv.integral(img).shape
    original = cv.integral(img)[pos[0]-1, pos[1]-1]/(img.shape[0] * img.shape[1])

    pos = cv.integral(g).shape
    gaus = cv.integral(g)[pos[0]-1, pos[1]-1]/(g.shape[0] * g.shape[1])

    pos = cv.integral(m).shape
    medi = cv.integral(m)[pos[0]-1, pos[1]-1]/(m.shape[0] * m.shape[1])

    pos = cv.integral(b).shape
    bila = cv.integral(b)[pos[0]-1, pos[1]-1]/(b.shape[0] * b.shape[1])

    print("Distance from Original to Gaussian: ", abs(original - gaus))
    print("Distance from Original to Median: ", abs(original - medi))
    print("Distance from Original to Bilateral: ", abs(original - bila))

    #Find lowest distance



#    =========================================================================
#    ==================== Task 8 =================================
#    =========================================================================
    print('Task 8:');
    img6 = cv.imread(img_path)
    intensity = cv.cvtColor(img6, cv.COLOR_BGR2HSV)[:,:,2]
    kernel1 = np.array([[0.0113, 0.0838, 0.0113],[0.0838,0.6193,0.0838],[0.0113,0.0838,0.0113]])
    kernel2 = np.array([[-0.8984, 0.1472, 1.1410],[-1.9075,0.1566,2.1359],[-0.8659,0.0573,1.0337]])

    ##a
    img_a_1 = cv.filter2D(intensity, -1, kernel1)
    img_a_2 = cv.filter2D(intensity, -1, kernel2)

    #b
    ##returns filtered image
    def decompose_and_filter(img, kernel_name, kernel):
        sigmas,u,vt = cv.SVDecomp(kernel)
        vec_index = 0
        if(sigmas[0][0]!=0):
            print("%s could not be decomposed ... approximating" % kernel_name)
            vec_index = np.where(sigmas==sigmas.max())[0][0]
        kernel1_V = np.sqrt(sigmas[vec_index,0])*u[:,vec_index]
        kernel1_H = np.sqrt(sigmas[vec_index,0])*vt[vec_index,:]
        return cv.sepFilter2D(img, -1, kernel1_H, kernel1_V)



    img_b_1 = decompose_and_filter(intensity, "kernel1", kernel1)
    img_b_2 = decompose_and_filter(intensity, "kernel2", kernel2)
    cv.imshow("T6: k1, 2d filt", img_a_1)
    cv.imshow("T6: k1, 1d filt", img_b_1)
    cv.imshow("T6: k2, 2d filt", img_a_2)
    cv.imshow("T6: k2, 1d filt", img_b_2)

    ##calculating differences
    delta_kernel1 = np.abs(img_a_1 - img_b_1)
    delta_kernel2 = np.abs(img_a_2 - img_b_2)
    text = "maximum pixel error for %s is %d"
    print(text % ("kernel1", delta_kernel1.max()))
    print(text % ("kernel2", delta_kernel2.max()))





    cv.waitKey(0)
    cv.destroyAllWindows()

