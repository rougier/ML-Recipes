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
        centroid = X.mean(axis = 0)
        Xc = X - centroid
        G = Xc @ Xc.T

        # Compute the eigenvalue/eigenvector decomposition of G
        eigvals, eigvecs = np.linalg.eigh(G)

        # It might be that some eigvals < 0
        # which theoretically cannot happen
        eigvals[eigvals < 0] = 0


        # TODO : handle when eigvals = 0
        # Compute the eigenvectors of the covariance matrix
        sig_eigvecs = (Xc.T @ eigvecs) / np.sqrt(eigvals)

        return eigvals, sig_eigvecs


    def transform(self, Z):
        '''
        TODO
        '''
        pass

if __name__ == '__main__':

    # Example 1 : 400 x (92, 112) Olivetti face dataset
    # We have much more dimensions (10304) than samples (400), a situation
    # where computing the PCA from the Gramm matrix is advantageous
    # -------------------------------------------------------------------------
    print("AT&T faces dataset example")
    print('-' * 70)

    samples = np.load('att_faces.npy')
    X, y = samples['input'], samples['label']

    ## Extract the 10 first principal component vectors
    n_components = 10
    eigenface = EigenFace(n_components = n_components)
    eigenface.fit(X)

