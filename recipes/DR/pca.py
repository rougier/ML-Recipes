# -----------------------------------------------------------------------------
# Copyright 2019 (C) Jeremy Fix
# Released under a BSD two-clauses license
#
# Reference: Pearson, K. (1901). "On Lines and Planes of Closest Fit to Systems
#            of Points in Space". Philosophical Magazine. 2 (11): 559–572.
#            doi:10.1080/14786440109462720
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

class PCA:
    ''' PCA class '''

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        '''
        Performs the PCA of X and stores the principal components.
        The datapoints are supposed to be stored in the row vectors of X.
        It keeps only the n_components projection vectors, associated
        with the n_components largest eigenvalues of the covariance matrix X.T X
        '''

        # Center the datapoints
        self.centroid = np.mean(X, axis=0)

        # Computes the covariance matrix
        sigma = np.dot((X - self.centroid).T, X - self.centroid)

        # Compute the eigenvalue/eigenvector decomposition of sigma
        eigvals, eigvecs = np.linalg.eigh(sigma)

        # Note :The eigenvalues returned by eigh are ordered in ascending order.

        # Stores the n_components eigvenvectors/eigenvalues associated
        # with the largest eigen values
        self.eigvals = eigvals[-self.n_components:]
        self.eigvecs = eigvecs[:, -self.n_components:]

    def transform(self, Z):
        '''
        Uses a fitted PCA to transform the row vectors of Z
        Remember the eigen vectors are ordred by ascending order
        Denoting Z_trans = transform(Z),
        The first  component is Z_trans[:, -1]
        The second component is Z_trans[:, -2]
        ...
        '''
        print((Z-self.centroid).shape, self.eigvecs.shape)
        return np.dot(Z - self.centroid, self.eigvecs)


if __name__ == '__main__':

    N = 500
    # Generate some random normal data rotated and translated
    X = np.column_stack([np.random.normal(0.0, 1.0, (N, )),
                         np.random.normal(0.0, 0.2, (N, ))])
    X = np.dot(X, np.array([[np.cos(1.0), np.sin(1.0)],[-np.sin(1.0), np.cos(1.0)]]))
    X = np.array([0.5, 1.0]) + X

    # Computes the PCA
    pca = PCA(n_components=2)
    pca.fit(X)

    X_trans = pca.transform(X)

    # Plot
    plt.figure()
    plt.scatter(X[:,0], X[:,1])
    # First axis
    plt.plot([pca.centroid[0]-2*pca.eigvecs[0, -1], pca.centroid[0]+2*pca.eigvecs[0,-1]],
             [pca.centroid[1]-2*pca.eigvecs[1, -1], pca.centroid[1]+2*pca.eigvecs[1,-1]],
             'g', linewidth=3)
    # Second axis
    plt.plot([pca.centroid[0]-2*pca.eigvecs[0, 0], pca.centroid[0]+2*pca.eigvecs[0,0]],
             [pca.centroid[1]-2*pca.eigvecs[1, 0], pca.centroid[1]+2*pca.eigvecs[1,0]],
             'r', linewidth=3)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.gca().set_aspect('equal')


    plt.figure()
    plt.scatter(X_trans[:, -1], X_trans[:, -2])

    plt.show()
