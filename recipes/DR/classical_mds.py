# -----------------------------------------------------------------------------
# Copyright 2020 (C) Jeremy Fix
# Released under a BSD two-clauses license
#
# Reference: W.S. Torgerson (1952)
#            Multidimensional scaling: I. Theory and method.
#            Psychometrika, 17: 401-419.
# Dataset : Database of faces from AT&T Laboratories Cambridge
#           https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np


class ClassicalMDS:

    def __init__(self, n_components):
        self.n_components = n_components

    def embed(self, delta):
        # Compute the Gram Matrix
        # by double centering the squared disimilarities
        n = delta.shape[0]
        H = np.eye(n) - 1/n * np.ones_like(d)
        G = -1./2. * H@(delta**2)@H

        # Compute the square root of the Gram matrix
        eigval, eigvec = np.linalg.eigh(G)

        if np.any(eigval < 0):
            print('''[WARNING] Some eigenvalues are non positive.
                  The computed reconstruction error only includes the positive
                  eigenvalues
                  ''')

        # It might be the eigenvalues are not all negative
        eigval[eigval < 0] = 0

        if np.any(eigval[-self.n_components:] < 0):
            print(f'''[ERROR] You want me to select {self.n_components}
                       but only {(eigval>0).sum()} are positive
                  ''')

        # The eigenvalues are ordered by increasing values
        # We keep the two largest
        embeddings = eigvec[:, -self.n_components:]
        eigvals = eigval[-self.n_components:]

        print(f'''Reconstruction error :
              {(1. - eigvals.sum()/eigval.sum())*100} %")''')

        for i in range(self.n_components):
            embeddings[:, i] *= np.sqrt(eigvals[i])

        return embeddings[:, ::-1], (1 - eigvals.sum()/eigval.sum())*100


if __name__ == '__main__':

    d = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]])

    mds = ClassicalMDS(2)
    embedding, error = mds.embed(d)

    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.show()
