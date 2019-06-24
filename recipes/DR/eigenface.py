# -----------------------------------------------------------------------------
# Copyright 2019 (C) Jeremy Fix
# Released under a BSD two-clauses license
#
# Reference: M. Turk & A. Pentland (1991) Eigenfaces for Recognition.
#            Journal of cognitive neuroscience, 3(1): 71-86.
# Dataset : Database of faces from AT&T Laboratories Cambridge
#           https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
# -----------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cmx
import matplotlib.colors as colors

class EigenFace:
    ''' EigenFace class '''

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        '''
        Compute the PCA by diagonalizing the Gram matrix
        The data are supposed to be laid out in the rows of X
        '''

        # Compute the Gram matrix, i.e. the matrix of
        # dot product of the centered samples
        self.centroid = X.mean(axis = 0)
        Xc = X - self.centroid
        G = Xc @ Xc.T

        # Compute the eigenvalue/eigenvector decomposition of G
        self.eigvals, self.eigvecs = np.linalg.eigh(G)

        # It might be that some eigvals < 0
        # which theoretically cannot happen
        self.eigvals[self.eigvals < 0] = 0

        # Compute the eigenvectors of the covariance matrix
        normalization = np.sqrt(self.eigvals)
        normalization[normalization == 0] = 1
        self.eigvecs = (Xc.T @ self.eigvecs) / normalization

    def transform(self, Z):
        '''
        Uses a fitted PCA to transform the row vectors of Z
        Remember the eigen vectors are ordred by ascending order
        Denoting Z_trans = transform(Z),
        The first  component is Z_trans[:, -1]
        The second component is Z_trans[:, -2]
        The coordinates in the projected space of the i-th sample
        is Z_trans[i, :]
        '''
        return (Z - self.centroid) @ self.eigvecs[:, -self.n_components:]

if __name__ == '__main__':

    # Example 1 : 400 x (112, 92) Olivetti face dataset
    # We have much more dimensions (10304) than samples (400), a situation
    # where computing the PCA from the Gramm matrix is advantageous
    # -------------------------------------------------------------------------
    print("AT&T faces dataset example")
    print('-' * 70)

    samples = np.load('att_faces.npy')
    X, y = samples['input'], samples['label']

    ## Extract the 15 first principal component vectors
    n_components = 10
    eigenface = EigenFace(n_components = n_components)
    eigenface.fit(X)

    ## Project the original data
    X_trans = eigenface.transform(X)

    print("{:.2f}% of the variance is kept with {} components".format(
          100 * eigenface.eigvals[-n_components:].sum()/eigenface.eigvals.sum(),
          n_components))

    ## Plot
    fig = plt.figure(figsize=(10,4), tight_layout=True)
    gs  = GridSpec(3, n_components)

    ### Coordinates in the projected space
    ### the color shows how the digits get separated by the principal vectors
    ax = fig.add_subplot(gs[0:2, :])
    ax.scatter(X_trans[:, -1], X_trans[:,-2] , alpha = .25)

    ### Projection vectors
    for i in range(n_components):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(eigenface.eigvecs[:, -(i+1)].reshape((112, 92)), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    ### Show some samples ordered by 1st principal component
    idx_sorted_by_1st_pc = np.argsort(X_trans[:,-1])
    plt.savefig('eigenface.png')

    ### Select a subset of equally spaced images ordered by increasing
    ### first PC
    selected_idx = idx_sorted_by_1st_pc[::40]

    ### Plot
    fig, axes = plt.subplots(1, len(selected_idx), figsize=(10,2), tight_layout=True)
    for i, idx in enumerate(selected_idx):
        ax = axes[i]
        ax.imshow(X[idx, :].reshape((112, 92)), cmap='gray')
        ax.set_title('{:.0f}'.format(X_trans[idx, -1]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('Few samples ordered by 1st PC')

    plt.savefig('eigenface_samples_1st.png')
    plt.show()
