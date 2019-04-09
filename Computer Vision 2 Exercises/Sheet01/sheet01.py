import numpy as np
from numpy.linalg import inv

err = 0
err_dual = 0
opt_sigma = 0
opt_num_clusters = 0
loss = 0
accuracy = 0

##############################################################################################################
#Auxiliary functions for Regression
##############################################################################################################
#returns features with bias X (num_samples*(1+num_features)) and target values Y (num_samples*target_dims)
def read_data_reg(filename):
    data = np.loadtxt(filename)
    Y = data[:,:2]
    X = np.concatenate((np.ones((data.shape[0], 1)), data[:,2:]), axis=1)
    return Y, X

#takes features with bias X (num_samples*(1+num_features)) and target values Y (num_samples*target_dims)
#returns regression coefficients w ((1+num_features)*target_dims)
def lin_reg(X, Y):
    Xt = X.T

    phi = np.linalg.inv(Xt.dot(Xt.T)).dot(Xt).dot(Y)

    w = phi
    return w

#takes features with bias X (num_samples*(1+num_features)), target Y (num_samples*target_dims) and regression coefficients w ((1+num_features)*target_dims)
#returns fraction of mean square error and variance of target prediction separately for each target dimension
def test_lin_reg(X, Y, w):
    Xt = X #.T
    
    var_0 = np.sum(np.power(Y[:,0] - np.mean(Y[:,0]),2))/Y.shape[0]
    var_1 = np.sum(np.power(Y[:,1] - np.mean(Y[:,1]),2))/Y.shape[0]

    y_0_est = w[:,0].dot(Xt)
    y_1_est = w[:,1].dot(Xt)

    mse_0 = np.mean(np.power(Y[:,0]-y_0_est,2))
    mse_1 = np.mean(np.power(Y[:,1]-y_1_est,2))

    err_0 = mse_0/var_0
    err_1 = mse_1/var_1

    err = [err_0, err_1]

    return err

#takes features with bias X (num_samples*(1+num_features)), centers of clusters C (num_clusters*(1+num_features)) and std of RBF sigma
#returns features mapped to higher embedding space (num_samples*num_clusters)
def RBF_embed(X, C, sigma):
    X_embed = np.zeros((X.shape[0],C.shape[0]))
    for i in range(X.shape[0]):
        for j in range(C.shape[0]):
            d = X[i,:]-C[j,:]
            X_embed[i,j] = np.exp(-0.5*np.dot(d, d)/(np.power(sigma,2)))
    return X_embed
############################################################################################################
#Linear Regression
############################################################################################################

def run_lin_reg(X_tr, Y_tr, X_te, Y_te):
    print('MSE/Var linear regression')

    w = lin_reg(X_tr, Y_tr)

    err = test_lin_reg(X_te.T, Y_te, w)
    print(err)
############################################################################################################
#Dual Regression
############################################################################################################
def getError(Y_hat, Y):
    var_y = np.var(Y,axis=0)
    mse = np.mean(np.power(Y-Y_hat,2), axis=0)
    err = mse/var_y
    
    return err

def computeK(Xi, Xj, lmbda):
    m = Xi.shape[0]
    n = Xj.shape[0]
    K = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            d = Xi[i,:]-Xj[j,:]
            K[i,j] = np.exp(-0.5*np.dot(d, d)/(np.power(lmbda,2)))
    return K

def run_dual_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list):
    X_tr_t = X_tr[tr_list]
    Y_tr_t = Y_tr[tr_list]
    X_tr_v = X_tr[val_list]
    Y_tr_v = Y_tr[val_list]
    
    sigma_opt = -1
    err_opt = [np.inf, np.inf]
    
    for sigma_pow in range(-5, 3):
        sigma = np.power(3.0, sigma_pow)
        
        K = computeK(X_tr_t, X_tr_t, sigma)
        Kinv = np.linalg.inv(K)
        k = computeK(X_tr_v, X_tr_t, sigma)
        
        Y_hat = np.matmul(k,Kinv).dot(Y_tr_t)
        
        err_hold = getError(Y_hat, Y_tr_v)
        print('MSE/Var dual regression for val sigma='+str(sigma))
        print(err_hold)
        
        if (np.mean(err_hold) < np.mean(err_opt)):
            err_opt = err_hold
            sigma_opt = sigma
            
    print("optimal sigma="+str(sigma_opt))
    print("optimal error="+str(err_opt))
    
    print('MSE/Var dual regression for test sigma='+str(sigma_opt))
    k = computeK(X_te, X_tr_t, sigma_opt)
    K = computeK(X_tr_t, X_tr_t, sigma_opt)
    Kinv = np.linalg.inv(K)
    Y_hat = np.matmul(k,Kinv).dot(Y_tr_t)
    print(getError(Y_hat, Y_te))


############################################################################################################
#Non Linear Regression
############################################################################################################
# problem 2 part 3
def run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list):
    X_tr_t = X_tr[tr_list]
    Y_tr_t = Y_tr[tr_list]
    X_tr_v = X_tr[val_list]
    Y_tr_v = Y_tr[val_list]
    
    sigma_opt = -1
    err_opt = [np.inf, np.inf]
    
    from sklearn.cluster import KMeans
    for num_clusters in [10, 30, 100]:
        kmeans = KMeans(num_clusters)
        kmeans.fit(X_tr_t)
        
        Y_centroid = np.vstack([np.mean(Y_tr_t[kmeans.labels_ == ind],axis=0) for ind in range(num_clusters)])
        for sigma_pow in range(-5, 3):
            sigma = np.power(3.0, sigma_pow)
            K = computeK(kmeans.cluster_centers_, kmeans.cluster_centers_, sigma)
            Kinv = np.linalg.inv(K)
            k = RBF_embed(X_tr_v, kmeans.cluster_centers_, sigma)
            Y_hat = np.matmul(k,Kinv).dot(Y_centroid)
            
            print('MSE/Var non linear regression for val sigma='+str(sigma)+' val num_clusters='+str(num_clusters))
            err_hold = getError(Y_hat, Y_tr_v)
            print(err_hold)
        
            if (np.mean(err_hold) < np.mean(err_opt)):
                num_clusters_opt = num_clusters
                err_opt = err_hold
                sigma_opt = sigma
    
    kmeans = KMeans(num_clusters_opt)
    kmeans.fit(X_tr_t)
    Y_centroid = np.vstack([np.mean(Y_tr_t[kmeans.labels_ == ind],axis=0) for ind in range(num_clusters_opt)])
    K = computeK(kmeans.cluster_centers_, kmeans.cluster_centers_, sigma_opt)
    Kinv = np.linalg.inv(K)
    k = RBF_embed(X_te, kmeans.cluster_centers_, sigma)
    Y_hat = np.matmul(k,Kinv).dot(Y_centroid)
    print("optimal sigma="+str(sigma_opt))
    print("optimal error="+str(err_opt))
    print("optimal cluster="+str(num_clusters_opt))
    print(X_te.shape)
    print(k.shape, Kinv.shape, Y_centroid.shape, Y_hat.shape, Y_te.shape)
    print('MSE/Var non-linear regression for test sigma='+str(sigma_opt)+' num of cluster=' + str(num_clusters_opt))
    print(getError(Y_hat, Y_te))

####################################################################################################################################
#Auxiliary functions for classification
####################################################################################################################################
#returns features with bias X (num_samples*(1+num_feat)) and gt Y (num_samples)
def read_data_cls(split):
    feat = {}
    gt = {}
    for category in [('bottle', 1), ('horse', -1)]: 
        feat[category[0]] = np.loadtxt('data/'+category[0]+'_'+split+'.txt')
        feat[category[0]] = np.concatenate((np.ones((feat[category[0]].shape[0], 1)), feat[category[0]]), axis=1)
        gt[category[0]] = category[1] * np.ones(feat[category[0]].shape[0])
    X = np.concatenate((feat['bottle'], feat['horse']), axis=0)
    Y = np.concatenate((gt['bottle'], gt['horse']), axis=0)
    return Y, X

def sig(w, X):
    z = np.matmul(X, w)
    return 1.0 / (1.0 + np.exp(-z))

#takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
#returns gradient with respect to w (num_features)
def log_llkhd_grad(X, Y, w):
    s = sig(w, X)
    dLdw = -np.matmul(X.T, s - (Y+1)/2)
    return dLdw

#takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
#returns log likelihood loss
def get_loss(X, Y, w):
    s = sig(w,X)
    loss = (np.dot((Y.T+1)/2, np.log(s)) + np.dot(1 - (Y.T+1)/2, np.log(1 - s)))
    return loss


#takes features with bias X (num_samples*(1+num_features)), gt Y (num_samples) and current_parameters w (num_features+1)
# Y must be from {-1, 1}
#returns accuracy
def get_accuracy(X, Y, w): 
    Y_hat = sig(w, X)
    Y_hat[Y_hat >= .5] = 1
    Y_hat[Y_hat < .5] = -1
    return np.count_nonzero(Y_hat == Y)/Y.shape[0]

####################################################################################################################################
#Classification
####################################################################################################################################
def run_classification(X_tr, Y_tr, X_te, Y_te, step_size):
    print('classification with step size '+str(step_size))
    max_iter = 10000
    w = np.random.rand(511)
    for step in range(max_iter):
        dLdw = log_llkhd_grad(X_tr, Y_tr, w)
        w = w + step_size*dLdw
        if step%1000 == 0:
            print("it=" + str(step) + " loss=" + str(get_loss(X_tr, Y_tr, w)) + " acc=" + str(get_accuracy(X_tr, Y_tr, w)))

    print('test set loss=' + str(get_loss(X_te, Y_te, w)) + ' accuracy='+str(get_accuracy(X_te, Y_te, w)))


####################################################################################################################################
#Exercises
####################################################################################################################################
Y_tr, X_tr = read_data_reg('data/regression_train.txt')
Y_te, X_te = read_data_reg('data/regression_test.txt')
print("#############Linear Regression###########")
run_lin_reg(X_tr, Y_tr, X_te, Y_te)

tr_list = list(range(0, int(X_tr.shape[0]/2)))
val_list = list(range(int(X_tr.shape[0]/2), X_tr.shape[0]-1))
print("#############Dual Regression###########")
run_dual_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)
print("#############Non-linear Regression###########")
run_non_lin_reg(X_tr, Y_tr, X_te, Y_te, tr_list, val_list)

print("#############Classification###########")
step_size = 1
Y_tr, X_tr = read_data_cls('test')
Y_te, X_te = read_data_cls('test')
run_classification(X_tr, Y_tr, X_te, Y_te, step_size)
