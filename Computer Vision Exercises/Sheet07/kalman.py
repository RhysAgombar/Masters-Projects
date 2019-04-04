import numpy as np
import matplotlib.pylab as plt

observations = np.load('observations.npy')

def get_observation(t):
    return observations[t]

class KalmanFilter(object):
    def __init__(self, Lambda, sigma_p, Phi, sigma_m):
        self.Lambda = Lambda ## Psi?
        self.sigma_p = sigma_p
        self.Phi = Phi
        self.sigma_m = sigma_m
        self.state = None
        self.convariance = None

    def init(self, init_state):
        self.state = init_state
        self.convariance = np.eye(init_state.shape[0]) * 0.01

    def track(self, xt):


        #self.state contains...  x, y, vx, vy

        mu_p = np.array([self.state[2], self.state[3], 0, 0]) #vx, vy, ax, ay
        mu_m = np.array([self.state[2], self.state[3], 0, 0])
        #mu_m = np.array([self.state[2], self.state[3]])

        mu_plus = mu_p + self.state #np.dot(self.Lambda, self.state)

        cov_plus = self.sigma_p + np.dot(np.dot(self.Lambda, self.convariance), self.Lambda.T)

        K =  np.dot(np.dot(cov_plus, self.Phi.T), np.linalg.inv(self.sigma_m + np.dot(np.dot(self.Phi, cov_plus), self.Phi.T)))

        xtt = np.array([xt[0],xt[1],0,0])
        mu_t = mu_plus + np.dot(K, (xtt - mu_m - np.dot(self.Phi, mu_plus)))

        cov_t = np.dot((np.identity(4) - np.dot(K, self.Phi)), cov_plus)

        self.state = mu_t
        self.convariance = cov_t

        pass

    def get_current_location(self):
        return self.Phi @ self.state


def main():
    init_state = np.array([0, 1, 0, 0])

    Lambda = np.array([[1, 0, 1, 0], # add 1s in here somewhere for velocity impact
                       [0, 1, 0, 1],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]) ## Psi?

    sp = 0.01
    sigma_p = np.array([[sp, 0, 0, 0],
                        [0, sp, 0, 0],
                        [0, 0, sp * 4, 0],
                        [0, 0, 0, sp * 4]])

    Phi = np.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    sm = 0.05
    sigma_m = np.array([[sm, 0, 0, 0],
                        [0, sm, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]]) ### figure out why this doesn't work as the basic 2x2



    #sigma_m = np.array([[sm, 0], [sm, 0]])

    tracker = KalmanFilter(Lambda, sigma_p, Phi, sigma_m)
    tracker.init(init_state)

    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())

    plt.figure()
    plt.plot([x[0] for x in observations], [x[1] for x in observations])
    plt.plot([x[0] for x in track], [x[1] for x in track])
    plt.show()


if __name__ == "__main__":
    main()




'''
mu_p = np.array([xt[0] - self.state[0], xt[1] - self.state[1], vx - self.state[2], vy - self.state[3]])
mu_m = np.array([xt[0], xt[1], vx, vy])

wt = mu_p + np.dot(self.Lambda, self.state)# + self.sigma_p


test = mu_p + np.dot(self.Lambda, self.state)

#print(test)
#print(test.shape)

Pr_wtwto = mu_p[0:2] + np.dot(self.Lambda[0:2,0:2], self.state[0:2]) #, self.sigma_p[0:2,0:2] #Norm(

Pr_xtwt = mu_m[0:2] + np.dot(self.Phi[0:2, 0:2], wt[0:2]) #, self.sigma_m




t = multivariate_gaussian(pos, Pr_wtwto, self.sigma_p[0:2,0:2])
t2 = multivariate_gaussian(pos, Pr_xtwt, self.sigma_p[0:2,0:2])

#cset = plt.contourf(X, Y, t + t2, zdir='z', offset=-0.15, cmap=cm.viridis)
#plt.show()


#x_pred = mu + np.dot(self.Lambda, xt) + self.sigma_p

#sig_pred = self.sigma_p + np.dot(np.dot(self.Lambda, SIGMA),self.Lambda.T)

#K = np.dot(np.dot(sig_pred,self.Phi.T), np.linalg.inv(self.sigma_m + np.dot(np.dot(self.Phi, sig_pred), self.Phi.T)))
'''
