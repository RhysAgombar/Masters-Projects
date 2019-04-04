import numpy as np
import utils

# ======================= PCA =======================
def pca(covariance, preservation_ratio=0.9):

    vals, vecs = np.linalg.eig(covariance)

    ind = np.argsort(vals)[::-1]
    vals = vals[ind]
    vecs = vecs[:,ind]

    total = np.sum(vals)
    threshold = total * 0.9

    count = 0
    for i in vals:
        count += 1
        total = total - i
        if total < threshold:
            break;

    vecs = vecs[:, :count]

    return vecs


# ======================= Covariance =======================

def dotFunc(sub):
    return np.dot(sub,sub.T)

def create_covariance_matrix(kpts, mean_shape):
    N = kpts.shape[0]

    sub = (kpts - mean_shape)

    cov = np.array(list(map(lambda x: dotFunc(x), sub[:]))) ## Not a loop!
    cov = np.sum(cov, axis=0)
    cov = cov / N

    return cov


# ======================= Visualization =======================

def visualize_impact_of_pcs(mean, pcs, pc_weights):
    # your part here
    pass


# ======================= Training =======================
def train_statistical_shape_model(kpts):
    mean_shape = np.mean(kpts, axis=0)
    mean_shape = mean_shape.reshape((1,mean_shape.shape[0], mean_shape.shape[1]))

    cov = create_covariance_matrix(kpts, mean_shape)
    pcs = pca(cov)

    stock_weights = np.zeros(pcs[0].shape)
    stock_weights.fill(1)
    neg_weights = np.zeros(pcs[0].shape)
    neg_weights.fill(-0.5)
    pos_weights = np.zeros(pcs[0].shape)
    pos_weights.fill(0.5)

    reconstruct_test_shape(kpts, mean_shape, pcs, stock_weights, "Reconstruction, Weights = 1")
    reconstruct_test_shape(kpts, mean_shape, pcs, neg_weights, "Reconstruction, Weights = -0.5")
    reconstruct_test_shape(kpts, mean_shape, pcs, pos_weights, "Reconstruction, Weights = 0.5")
    utils.visualize_hands(kpts, "KPTS Values", delay=0.1, ax=None, clear=False)
    utils.visualize_hands(mean_shape, "Mu", delay=0.1, ax=None, clear=False)

    return mean_shape, pcs, stock_weights

    pass

# ======================= Reconstruct =======================

#Remember, you guys asked for this.
def rconstFunc(kpts, mean, pcs, pc_weight, pos):
    return np.squeeze(mean + np.dot(pcs, np.dot(pcs.T, np.squeeze(kpts[pos] - mean)) * pc_weight[:, np.newaxis]))

def reconstruct_test_shape(kpts, mean, pcs, pc_weight, title=""):

    pos = np.array(range(kpts.shape[0]))
    reconst = np.array(list(map(lambda x: rconstFunc(kpts, mean, pcs, pc_weight, x), pos[:])))  ## Not a loop!

    utils.visualize_hands(reconst, title, delay=0.1, ax=None, clear=False)

    return reconst
