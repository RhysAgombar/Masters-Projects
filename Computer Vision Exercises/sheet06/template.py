#!/usr/bin/python3.5

import numpy as np
import cv2 as cv
from scipy.stats import multivariate_normal

'''
    read the usps digit data
    returns a python dict with entries for each digit (0, ..., 9)
    dict[digit] contains a list of 256-dimensional feature vectores (i.e. the gray scale values of the 16x16 digit image)
'''
def read_usps(filename):
    data = dict()
    with open(filename, 'r') as f:
        N = int( np.fromfile(f, dtype = np.uint32, count = 1, sep = ' ') )
        for n in range(N):
            c = int( np.fromfile(f, dtype = np.uint32, count = 1, sep = ' ') )
            tmp = np.fromfile(f, dtype = np.float64, count = 256, sep = ' ') / 1000.0
            data[c] = data.get(c, []) + [tmp]
    for c in range(len(data)):
        data[c] = np.stack(data[c])
    
    return data

'''
    load the face image and foreground/background parts
    image: the original image
    foreground/background: numpy arrays of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
'''
def read_face_image(filename):
    image = misc.imread(filename) / 255.0
    bounding_box = np.zeros(image.shape)
    bounding_box[50:100, 60:120, :] = 1
    foreground = image[bounding_box == 1].reshape((50 * 60, 3))
    background = image[bounding_box == 0].reshape((40000 - 50 * 60, 3))
    return image, foreground, background



def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

'''
    implement your GMM and EM algorithm here
'''
class GMM(object):
    GMM_L = []
    GMM_mu = []
    GMM_var = []

    '''
        fit a single gaussian to the data
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
    '''
    def fit_single_gaussian(self, data):
        d = data.shape[1]
        N = data.shape[0] ##number of observations
        mu = np.mean(data, axis=0) # mean vector
        mu.shape = (1,d)
        var = []
        for pos in range(0, d):

            var_d = sum( (data[:,pos] - mu[0,pos])**2 )/N # variance
            var.append(var_d)
        var = np.diag(var)



        sz = 256
        x_values = np.linspace(0.00001, 1, sz)
        test = np.zeros((sz,d))

#        for pos in range(0, d): #20 #101
##            x = np.linspace(0,N, sz)
#            test[:,pos] = gaussian(x_values, mu[0,pos], var[pos,pos])

        #plt.imshow(test)
        #plt.show()
        L = np.ones((1,1)) ##normalizing
        print(L.shape)
        print(mu.shape)
        print(var.shape)

        self.GMM_L = L
        self.GMM_mu = mu
        self.GMM_var = var
        self.GMM_var.shape = (1,d,d)





    def Dnormal(self,x,h):
        D = self.GMM_var.shape[1]
#        cov_inv =  np.divide(np.diag([1]*D), self.GMM_var[h], out=np.zeros((D,D)), where=self.GMM_var[h]!=0)

        ##calculating determinant
        det = 1
#        for di in range(D):det *= self.GMM_var[h][di][di]
        det_cov = 1
        cov_inv = self.GMM_var ##we are assuming that covariance matrix is the unitary matrix
        
        diff = x - self.GMM_mu[h]
        
        diff.shape = (1, D)
        ##calculates density
        #result =np.matmul( np.matmul(diff, cov_inv),diff.transpose())
        #print(result)
        #result = np.exp(-0.5*result)
        #print(result)
        #result = result/np.sqrt( (2*np.pi)**D * det_cov )
        #return np.log(result[0,0])

        ##calculates likelihood function
        result =-0.5*(np.log(det_cov) -np.matmul( np.matmul(diff, cov_inv),diff.transpose())[0,0] - D*np.log(np.pi))
        return result

        
        '''
        implement the em algorithm here
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
        @n_iterations: the number of em iterations
        '''
    def em_algorithm(self, data, n_iterations = 10):
        
        I = data.shape[0] # number of observations
        K = self.GMM_mu.shape[0] ## number of gaussians
        D = data.shape[1] ## number of dimensions



        print(self.GMM_var.shape)
        print(self.GMM_mu.shape)

        #expectation
        r = np.zeros((I,K))
        #for each input instance calculate probability from k gaussians
        for h in range(K):
            nm = multivariate_normal(self.GMM_mu[h], self.GMM_var[h],allow_singular = True  )
            for i in range(I):
                r[i,h] = self.GMM_L[h]* np.log(nm.pdf(data[i]))
                #r[i,h] = self.GMM_L[h]* self.Dnormal(data[i], h)
                    ##maybe vectorize this?
        #normalizing l
        ##sum over h
        su = r.sum(axis=1)
        su.shape = (I, 1)
        r = r / su
            
        #maximization
        #            for h in range(K):
        sum_r = r.sum()
        sum_r_over_i = r.sum(axis=0)
        sum_r_over_i.shape = (K,1)
            
        self.GMM_L = sum_r_over_i / sum_r
        self.GMM_mu = np.matmul(r.transpose(), data) / sum_r_over_i # KxD


        
        diagonal_mask = np.diag([1]*D)
        #for each gaussian we will calculate a matrix
        for h in range(K):
            sigma = np.zeros((D,D))
            for i in range(I):
                diff_x_mu = data[i] - self.GMM_mu[h] ##results in a 1xD matrix
                diff_x_mu.shape=(D,1)
                diff_x_mu = np.matmul(diff_x_mu.transpose(), diff_x_mu) ## DxD matrix
                sigma += r[i,h]*diff_x_mu
            sigma = sigma / sum_r_over_i[h]
            ##get only diagonal
            sigma *= diagonal_mask
            self.GMM_var[h] = sigma
        
        
        '''
        implement the split function here
        generates an initialization for a GMM with 2K components out of the current (already trained) GMM with K components
        @epsilon: small perturbation value for the mean
        '''
    def split(self, epsilon = 0.1):
        ##duplicating lambdas and normalizing values
        self.GMM_L = np.concatenate((self.GMM_L,self.GMM_L),axis=0) / 2.0
        #recalculating means

        H = self.GMM_mu.shape[0] ##number of gaussians
        u2 = np.copy(self.GMM_mu)
        for h in range(H):
            var = np.diag(self.GMM_var[h])
            self.GMM_mu[h] = self.GMM_mu[h] + (epsilon*var)
            u2[h] = self.GMM_mu[h] - epsilon*var

        
        self.GMM_mu = np.concatenate((self.GMM_mu,u2), axis=0)
        
        #duplicating var
        self.GMM_var = np.concatenate((self.GMM_var, self.GMM_var),axis=0)
       
        print(np.isnan(self.GMM_var).any())
        print(np.isnan(self.GMM_mu).any())
        print(np.isnan(self.GMM_L).any())
    




        '''
        sample a D-dimensional feature vector from the GMM
        '''
    def sample(self):
        #TODO
        fea = np.zeros((1,self.GMM_mu.shape[1])) #number of dimensions

        print("chekcing for nan or inf")
        print(np.isnan(self.GMM_var).any())
        print(np.isnan(self.GMM_mu).any())
        print(np.isnan(self.GMM_L).any())
        print(not np.isfinite(self.GMM_var).any())
        print(not np.isfinite(self.GMM_mu).any())
        print(not np.isfinite(self.GMM_L).any())

        if(np.isnan(self.GMM_var).any()):
            print("var:", self.GMM_var)
            print("mu:", self.GMM_mu)
            print("L:", self.GMM_L)
    
        
        for h in range(self.GMM_mu.shape[0]): ##for each gaussian
            fea += self.GMM_L[h]* np.random.multivariate_normal(self.GMM_mu[h], self.GMM_var[h])
        
        return fea



        '''
        Task 2d: synthesizing handwritten digits
        if you implemeted the code in the GMM class correctly, you should not need to change anything here
        '''
data = read_usps('usps.txt')

gmm = [ GMM() for _ in range(10) ] # 10 GMMs (one for each digit)
for split in [0, 1, 2]:
    print("MU AT ITER",gmm[0].GMM_mu)
    result_image = np.zeros((160, 160))
    for digit in range(10):
        # train the model
        if split == 0:
            gmm[digit].fit_single_gaussian(data[digit])
        else:
            gmm[digit].em_algorithm(data[digit])
        # sample 10 images for this digit
        for i in range(10):
            x = gmm[digit].sample()
            #print("sample", x)
            x = x.reshape((16, 16))
            x = np.clip(x, 0, 1)*255
            x = x.astype(np.uint8)
            result_image[digit*16:(digit+1)*16, i*16:(i+1)*16] = x
        # save image
        cv.imwrite('digits.' + str(2 ** split) + 'components.png', result_image)
        # split the components to have twice as many in the next run
        gmm[digit].split(epsilon = 0.1)

#gmm = GMM()
#gmm.fit_single_gaussian(data[0])

#gmm.em_algorithm(data[0])
#gmm.split()
#gmm.em_algorithm(data[0])


'''
    Task 2e: skin color model
'''
#image, foreground, background = read_face_image('face.jpg')

'''
    TODO: compute p(x|w=foreground) / p(x|w=background) for each image pixel and manipulate image such that everything below the threshold is black, display the resulting image
'''
