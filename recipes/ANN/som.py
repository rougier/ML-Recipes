# -----------------------------------------------------------------------------
# Copyright 2019 (C) Nicolas P. Rougier
# Released under a BSD two-clauses license
#
# References: Kohonen, Teuvo. Self-Organization and Associative Memory.
#             Springer, Berlin, 1984.
# -----------------------------------------------------------------------------
import numpy as np


class SOM:
    """ Self Organizing Map """

    def __init__(self, shape, distance):
        ''' Initialize som '''

        self.codebook = np.random.uniform(0, 1, shape)
        self.labels = np.random.uniform(0, 1, len(self.codebook))
        self.distance = distance / distance.max()
        
        
    def learn(self, samples,
              n_epoch=10000, sigma=(0.25, 0.01), lrate=(0.5, 0.01)):
        """ Learn samples """

        t = np.linspace(0,1,n_epoch)
        lrate = lrate[0]*(lrate[1]/lrate[0])**t
        sigma = sigma[0]*(sigma[1]/sigma[0])**t
        I = np.random.randint(0, len(samples), n_epoch)
        samples = samples[I]

        for i in range(n_epoch):
            # Get random sample
            data = samples[i]

            # Get index of nearest node (minimum distance)
            winner = np.argmin(((self.codebook - data)**2).sum(axis=-1))

            # Gaussian centered on winner
            G = np.exp(-self.distance[winner]**2/sigma[i]**2)

            # Move nodes towards sample according to Gaussian 
            self.codebook -= lrate[i]*G[...,np.newaxis]*(self.codebook - data)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import scipy.spatial
    import matplotlib.pyplot as plt

    n = 16
    X,Y = np.meshgrid(np.linspace(0, 1, n),np.linspace(0, 1, n))
    P = np.c_[X.ravel(), Y.ravel()]
    D = scipy.spatial.distance.cdist(P,P)
    som = SOM((len(P),2), D)
    
    T = np.random.uniform(0.0, 2.0*np.pi, 25000)
    R = np.sqrt(np.random.uniform(0.50**2, 1.0**2, len(T)))
    samples = np.c_[R*np.cos(T), R*np.sin(T)]

    som.learn(samples, 25000, sigma=(0.50, 0.01), lrate=(0.50, 0.01))

    # Draw result
    fig = plt.figure(figsize=(8,8))
    axes = fig.add_subplot(1,1,1)

    # Draw samples
    x,y = samples[:,0], samples[:,1]
    plt.scatter(x, y, s=1.0, color='b', alpha=0.1, zorder=1)
    
    # Draw network
    x,y = som.codebook[:,0].reshape(n,n), som.codebook[:,1].reshape(n,n)
    for i in range(n):
        plt.plot (x[i,:], y[i,:], 'k', alpha=0.85, lw=1.5, zorder=2)
        plt.plot (x[:,i], y[:,i], 'k', alpha=0.85, lw=1.5, zorder=2)
    plt.scatter (x, y, s=50, c='w', edgecolors='k', zorder=3)
    
    plt.axis([-1,1,-1,1])
    plt.xticks([]), plt.yticks([])
    plt.show()
