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

import urllib
import geocoder
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import yaml


class ClassicalMDS:

    def __init__(self, n_components):
        self.n_components = n_components

    def embed(self, delta):
        '''
        Arguments:
            delta: square matrix of dissimilarities

        Returns:
            embeddings : a N x d matrix of the embeddings in the rows
            error : the reconstruction error (in 0%-100%)
        '''
        # Compute the Gram Matrix
        # by double centering the squared disimilarities
        n = delta.shape[0]
        H = np.eye(n) - 1/n * np.ones_like(delta)
        G = -1./2. * H@(delta**2)@H

        # Compute the square root of the Gram matrix
        eigval, eigvec = np.linalg.eigh(G)

        if np.any(eigval < 0):
            print("[WARNING] Some eigenvalues are non positive."
                  " The computed reconstruction error only includes"
                  " the positive eigenvalues")

        # It might be the eigenvalues are not all negative
        eigval[eigval < 0] = 0

        if np.any(eigval[-self.n_components:] < 0):
            print(f"[ERROR] You want me to select {self.n_components}"
                  f" but only {(eigval>0).sum()} are positive")

        # The eigenvalues are ordered by increasing values
        # We keep the two largest
        embeddings = eigvec[:, -self.n_components:]
        eigvals = eigval[-self.n_components:]

        print("Reconstruction error : "
              f"{(1. - eigvals.sum()/eigval.sum())*100} %")

        for i in range(self.n_components):
            embeddings[:, i] *= np.sqrt(eigvals[i])

        return embeddings[:, ::-1], (1 - eigvals.sum()/eigval.sum())*100


def get_car_distances(cities):
    '''
    Arguments:
        cities: list of cities as str

    Returns:
        a matrix of car distances between each pair of cities
    '''
    locations = []
    for city in tqdm.tqdm(cities):
        g = geocoder.osm(f"{city}")
        locations.append(f"{g.osm['y']},{g.osm['x']}")
    cities_long_lat = ";".join(locations)

    url=f"http://router.project-osrm.org/table/v1/car/{cities_long_lat}?annotations=distance"

    yml_file = urllib.request.urlopen(url)
    yml_data = yaml.safe_load(yml_file)
    distances = np.array(yml_data['distances'])
    return distances


def get_bird_distances(cities):
    '''
    Arguments:
        cities: list of cities as str

    Returns:
        a matrix of geodesic distances between each pair of cities
    '''
    locations = []
    for city in tqdm.tqdm(cities):
        g = geocoder.osm(f"{city}")
        locations.append((g.osm['y'], g.osm['x']))

    distances = []
    for l1 in locations:
        d = []
        for l2 in locations:
            d.append(geodesic(l1, l2).kilometers)
        distances.append(d)

    return np.array(distances)

if __name__ == '__main__':

    # A dummy example
    d = np.array([[0, 1, 1], [1, 0, 2], [2, 0.5, 0]])

    mds = ClassicalMDS(2)
    embedding, error = mds.embed(d)

    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.show()

    # Example with cities distances and traveling distances
    cities = ['Paris, France', 'Lyon, France',
              'Lille, France', 'Strasbourg, France',
              'Toulon, France', 'Clermont-Ferrand, France',
              'Nancy, France', 'Orléans, France',
              'Koln, Germany', 'Frankfurt am Main, Germany',
              'Amsterdam, Netherlands', 'Brussel, Belgium']

