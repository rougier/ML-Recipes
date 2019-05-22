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
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cmx
import matplotlib.colors as colors

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
        RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


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
        #self.eigvals = eigvals[-self.n_components:]
        #self.eigvecs = eigvecs[:, -self.n_components:]

        # Stores all the eigenvectors/eigenvalues
        # Usefull for latter computing some statistics of the variance we keep
        self.eigvals = eigvals
        self.eigvecs = eigvecs

    def transform(self, Z):
        '''
        Uses a fitted PCA to transform the row vectors of Z
        Remember the eigen vectors are ordred by ascending order
        Denoting Z_trans = transform(Z),
        The first  component is Z_trans[:, -1]
        The second component is Z_trans[:, -2]
        ...
        '''
        return np.dot(Z - self.centroid, self.eigvecs[:, -self.n_components:])


if __name__ == '__main__':


    # Example 1 : Random 2D data
    # -------------------------------------------------------------------------
    print("Random data example")
    print('-' * 70)

    N = 500

    ## Generate some random normal data rotated and translated
    X = np.column_stack([np.random.normal(0.0, 1.0, (N, )),
                         np.random.normal(0.0, 0.2, (N, ))])
    X = np.dot(X, np.array([[np.cos(1.0), np.sin(1.0)],[-np.sin(1.0), np.cos(1.0)]]))
    X = np.array([0.5, 1.0]) + X

    ## Extract the 2 first principal component vectors
    ## With a 2D dataset, these are the only interesting components
    n_components = 2
    pca = PCA(n_components=n_components)
    pca.fit(X)

    ## Project the original data
    X_trans = pca.transform(X)

    print("{:.2f}% of the variance along the first axis (green)".format(
          100 * pca.eigvals[-1] / pca.eigvals.sum()))

    print("{:.2f}% of the variance along the second axis (red)".format(
          100 * pca.eigvals[-2] / pca.eigvals.sum()))

    ## Plot
    plt.figure()
    plt.scatter(X[:,0], X[:,1], alpha = .25)
    ## Plot the first projecton axis in green
    plt.plot([pca.centroid[0]-2*pca.eigvecs[0, -1], pca.centroid[0]+2*pca.eigvecs[0,-1]],
             [pca.centroid[1]-2*pca.eigvecs[1, -1], pca.centroid[1]+2*pca.eigvecs[1,-1]],
             'g', linewidth=3)
    ## Plot the second projection axis in red
    plt.plot([pca.centroid[0]-2*pca.eigvecs[0, 0], pca.centroid[0]+2*pca.eigvecs[0,0]],
             [pca.centroid[1]-2*pca.eigvecs[1, 0], pca.centroid[1]+2*pca.eigvecs[1,0]],
             'r', linewidth=3)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.gca().set_aspect('equal')
    plt.savefig("random_points.png")
    print()

    # Example 2 : 28 x 28 images of digits
    # -------------------------------------------------------------------------
    print("Digits example")
    print('-' * 70)

    samples = np.load('dig_app_text.cb.npy')
    X, y = samples['input'], samples['label']

    ## Extract the 10 first principal component vectors
    n_components = 10
    pca = PCA(n_components = n_components)
    pca.fit(X)

    ## Project the original data
    X_trans = pca.transform(X)

    print("{:.2f}% of the variance is kept with {} components".format(
          100 * pca.eigvals[-n_components:].sum()/pca.eigvals.sum(),
          n_components))

    ## Plot
    fig = plt.figure(figsize=(10,4), tight_layout=True)
    gs  = GridSpec(3, 10)

    ### Coordinates in the projected space
    ### the color shows how the digits get separated by the principal vectors
    ax = fig.add_subplot(gs[0:2, :-2])
    cmap = get_cmap(10)
    ax.scatter(X_trans[:, -1], X_trans[:,-2] , c=[cmap(l) for l in y], alpha = .25)
    ax = fig.add_subplot(gs[0:2, -2:])
    ax.set_axis_off()
    for i in range(10):
        col = cmap(i)
        rect = plt.Rectangle((0.1, i/10.), 0.4, 1.0/14., facecolor=col)
        ax.add_artist(rect)
        ax.text(0.6, i/10.+1./28., str(i), fontsize=12, color="k", **{'ha':'center', 'va':'center'})


    ### Projection vectors
    for i in range(n_components):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(pca.eigvecs[:, -(i+1)].reshape((28, 28)), cmap='gray_r')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig("digits.png")
    plt.show()
