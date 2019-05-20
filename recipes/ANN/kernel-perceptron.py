# -----------------------------------------------------------------------------
# Copyright 2019 (C) Jeremy Fix
# Released under a BSD two-clauses license
#
# References: Aizerman, M. A., Braverman, E. A. and Rozonoer, L.. " Theoretical
#             foundations of the potential function method in pattern
#             recognition learning.." Paper presented at the meeting of the
#             Automation and Remote Control,, 1964.
# The algorithm is described p. 825 though some kernels get introduced latter
# in history.
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def RBF_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y) / (2.0 * sigma**2))

def linear_kernel(x, y):
    return np.dot(x, y)

def poly_kernel(x, y, d):
    return (1 + np.dot(x, y))**d

class KernelPerceptron:

    def __init__(self, kernel):
        ''' Initialization of the perceptron with the given kernel'''
        self.kernel = kernel
        self.kernels = []
        self.coeffs = []

    def learn(self, input_sample, output_sample):
        ''' If the sample is misclassified, it brings in a new kernel '''
        if self(input_sample) != output_sample:
            self.kernels.append(lambda x, y=input_sample: self.kernel(x, y))
            self.coeffs.append(output_sample)


    def __call__(self, inpout):
        ''' The output is the sign of the weighted kernels '''
        v = 0.0
        for c, k in zip(self.coeffs, self.kernels):
            v += c * k(inpout) + c
        return -1 + 2 * (v > 0) # to return -1 or +1

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def train(samples, kernel, epochs=250):
        network = KernelPerceptron(kernel)
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.learn(samples['input'][n], samples['output'][n])
        return network

    def accuracy(network, samples):
        num_correct = 0
        for i in range(samples.size):
            o = network(samples['input'][i])
            num_correct += (o == samples['output'][i])
        return num_correct / float(samples.size)

    def make_moons(num_samples):
        pos_x = -0.5 + np.cos(np.linspace(0, np.pi, num=num_samples))
        pos_y =        np.sin(np.linspace(0, np.pi, num=num_samples))
        pos_samples = np.column_stack([pos_x, pos_y])

        neg_x = 0.5 + np.cos(np.linspace(np.pi, 2.0 * np.pi, num=num_samples))
        neg_y =       np.sin(np.linspace(np.pi, 2.0 * np.pi, num=num_samples))
        neg_samples = np.column_stack([neg_x, neg_y])

        samples = np.zeros(2*num_samples, dtype=[('input',  float, 2),
                                                 ('output', float, 1)])
        samples['input'] = np.vstack([pos_samples, neg_samples]) + \
                           np.random.normal(0.0, 0.25, (2*num_samples,2))
        samples['output'] = np.hstack([np.ones((num_samples,)),
                                      -np.ones((num_samples,))])
        return samples

    def make_blobs(num_samples):
        pos_x = np.random.normal(-1.0, 0.5, (num_samples,))
        pos_y = np.random.normal( 0.0, 0.5, (num_samples,))
        pos_samples = np.column_stack([pos_x, pos_y])

        neg_x = np.random.normal( 1.0, 0.5, (num_samples,1))
        neg_y = np.random.normal( 0.0, 0.5, (num_samples,1))
        neg_samples = np.hstack([neg_x, neg_y])

        samples = np.zeros(2*num_samples, dtype=[('input',  float, 2),
                                                 ('output', float, 1)])
        samples['input']  = np.vstack([pos_samples, neg_samples])
        samples['output'] = np.hstack([np.ones((num_samples,)),
                                      -np.ones((num_samples,))])
        return samples

	# Example 1 : Blobs
    # -------------------------------------------------------------------------
    print("Learning the blobs dataset with a polynomial kernel")
    train_samples = make_moons(50)
    test_samples  = make_moons(500)

    # Create an appropriate kernel
    # rbf = lambda x, y: RBF_kernel(x, y, 0.25)
    poly = lambda x, y: poly_kernel(x, y, 3)

    # Learn the classifier
    network = train(train_samples, poly)

    # Estimates the risks
    empirical_accuracy = accuracy(network, train_samples)
    real_accuracy = accuracy(network, test_samples)

    print("Empirical accuracy : {} %".format(100 * empirical_accuracy))
    print("Estimated real accuracy : {} %".format(100 * real_accuracy))

    # Plots
    ngrid = 250
    X, Y = np.meshgrid(np.linspace(-2, 2, ngrid),
                       np.linspace(-2, 2, ngrid))
    Z = np.zeros_like(X)
    for i in range(ngrid):
        for j in range(ngrid):
            Z[i, j] = network(np.array([X[i,j], Y[i,j]]))

    plt.figure()
    plt.plot(train_samples['input'][:50, 0], train_samples['input'][:50, 1],
             'ko', markerfacecolor='w', markersize=8, markeredgewidth=2)
    plt.plot(train_samples['input'][50:, 0], train_samples['input'][50:, 1],
             'k^', markerfacecolor='k', markersize=8, markeredgewidth=2)
    CS4 = plt.contour(X, Y, Z, [0],
                      colors = ('k',),
                      linewidths = (3,))

    CS = plt.contourf(X, Y, Z, [-2,0,2],
                      alpha=0.2,
                      cmap=plt.cm.bone)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])

    plt.show()
