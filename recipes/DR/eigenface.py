# -----------------------------------------------------------------------------
# Copyright 2019 (C) Jeremy Fix
# Released under a BSD two-clauses license
#
# Reference:  TODO
# Dataset : Olivetti faces TODO

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
        TODO
        '''
        pass


    def transform(self, Z):
        '''
        TODO
        '''
        pass

if __name__ == '__main__':

    # Example 1 : 400 x (64 , 64) Olivetti face dataset
    # We have much more dimensions (4096) than samples (400), a situation
    # where computing the PCA with eigenface is advantageous
    # -------------------------------------------------------------------------
    print("Olivetti faces example")
    print('-' * 70)

    samples = np.load('dig_app_text.cb.npy')
    X = samples['input']

    ## Extract the 10 first principal component vectors
    n_components = 10
    eigenface = EigenFace(n_components = n_components)
    eigenface.fit(X)


