# coding: utf-8
# -----------------------------------------------------------------------------
# Copyright 2020 (C) Jeremy Fix
# Released under a BSD two-clauses license
#
# Reference: W.S. Torgerson (1952)
#            Multidimensional scaling: I. Theory and method.
#            Psychometrika, 17: 401-419.
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
    # Request the longitudes and latitudes of the cities
    for city in tqdm.tqdm(cities):
        geocity = geocoder.osm(f"{city}")
        locations.append(f"{geocity.osm['y']},{geocity.osm['x']}")
    cities_long_lat = ";".join(locations)

    # Using osrm to get the travel distance by car
    url=f"http://router.project-osrm.org/table/v1/car/{cities_long_lat}?annotations=distance"
    yml_file = urllib.request.urlopen(url)
    yml_data = yaml.safe_load(yml_file)
    distances = np.array(yml_data['distances'])/1000.

    return distances


def get_geodesic_distances(cities):
    '''
    Arguments:
        cities: list of cities as str

    Returns:
        a matrix of geodesic distances between each pair of cities
    '''
    locations = []
    # Request the longitudes and latitudes of the cities
    for city in tqdm.tqdm(cities):
        geocity = geocoder.osm(f"{city}")
        locations.append((geocity.osm['y'], geocity.osm['x']))

    # Compute the geodesic distance between each pair of cities
    distances = np.zeros((len(cities), len(cities)))
    for i, l1 in enumerate(locations):
        for j, l2 in enumerate(locations[i+1:]):
            distances[i, i+j+1] = geodesic(l1, l2).kilometers
            distances[i+j+1, i] = distances[i, i+j+1]

    return distances


if __name__ == '__main__':

    mds = ClassicalMDS(n_components=2)

    # Example with cities distances and traveling distances
    cities = ['Paris, France', 'Lyon, France',
              'Lille, France', 'Strasbourg, France',
              'Toulon, France', 'Clermont-Ferrand, France',
              'Nancy, France', 'Orléans, France',
              'Koln, Germany', 'Frankfurt am Main, Germany',
              'Amsterdam, Netherlands', 'Brussel, Belgium']

    # Compute the embeddings from the geodesic distance
    # as the earth is "flat", we expect a low error
    geodesic_distances = get_geodesic_distances(cities)
    embeddings_geodesic, error_geodesic = mds.embed(geodesic_distances)

    car_distances = get_car_distances(cities)
    embeddings_car, error_car = mds.embed(car_distances)

    # Plot the embeddings
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    ax = axs[0]
    ax.scatter(embeddings_geodesic[:, 1], embeddings_geodesic[:, 0])
    for ci, c in enumerate(cities):
        ax.annotate(c.split(',')[0],
                    (embeddings_geodesic[ci, 1], embeddings_geodesic[ci, 0]),
                    xycoords="data", xytext=(0, 10),
                    textcoords='offset points', ha="center")
    ax.set_title(f"Geodesic. Reconstruction error {error_geodesic:4.2f}%")

    ax = axs[1]
    ax.scatter(embeddings_car[:, 1], embeddings_car[:, 0])
    for ci, c in enumerate(cities):
        ax.annotate(c.split(',')[0], (embeddings_car[ci, 1], embeddings_car[ci, 0]),
                    xycoords="data", xytext=(0, 10),
                    textcoords='offset points', ha="center")
    ax.set_title(f"Car. Reconstruction error {error_car:4.2f}%")

    plt.suptitle("Multidimensional scaling")
    plt.savefig("classical_mds.png", bbox_inches='tight')
    plt.show()
