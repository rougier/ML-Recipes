# -----------------------------------------------------------------------------
# Copyright 2019 (C) Jeremy Fix
# Released under a BSD two-clauses license
#
# References: Y. Freund, R. E. Schapire. "Large margin classification using
#             the perceptron algorithm". In: 11th Annual Conference on
#             Computational Learning Theory, New York, NY, 209-217, 1998.
#             DOI:10.1023/A:1007662407062
# Online algorithm of Fig. 1 in the cited reference
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

class VotedPerceptron:

    def __init__(self, n, m):
        ''' Initialization of the voted perceptron with given sizes.  '''
        self.n = n
        self.m = m
        self.input  = np.ones(n+1)
        self.output = np.ones(m)

        self.weights = np.zeros((1, self.m, self.n+1))
        self.votes = np.zeros((1,))

    def learn(self, input_sample, output_sample):
        ''' The learning function : a single sample is expected '''
        self.input[1:] = input_sample

        # Compute the prediction with the most recently created perceptron
        o = np.sign(np.dot(self.weights[-1,:,:], self.input))
        if o == output_sample:
            # Increase the confidence in this perceptron
            self.votes[-1] += 1
        else:
            # Create a new perceptron
            self.weights = np.vstack([self.weights, np.zeros((1, self.m, self.n+1))])
            self.weights[-1, ...] = self.weights[-2, ...] + output_sample * self.input
            self.votes   = np.vstack([self.votes  , 1])


    def __call__(self, input_sample):
        ''' Prediction step '''
        self.input[1:] = input_sample
        outputs = np.sign(np.dot(self.weights, self.input))
        return np.sign(np.sum(outputs * self.votes))

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def train(samples, epochs=250):
        network = VotedPerceptron(2,1)
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.learn(samples['input'][n], samples['output'][n])
        return network

    def test(network, samples):
        for i in range(samples.size):
            o = network(samples['input'][i])
            print(i, samples['input'][i], '%.2f' % o,
                  '(expected %.2f)' % samples['output'][i])

    def accuracy(network, samples):
        num_correct = 0
        for i in range(samples.size):
            o = network(samples['input'][i])
            num_correct += (o == samples['output'][i])
        return num_correct / float(samples.size)


    def make_blobs(num_samples):
        pos_samples = np.hstack([np.random.normal(-1.0, 0.5, (num_samples,1)),
                                 np.random.normal( 0.0, 0.5, (num_samples,1))])
        neg_samples = np.hstack([np.random.normal( 1.0, 0.5, (num_samples,1)),
                                 np.random.normal( 0.0, 0.5, (num_samples,1))])
        samples = np.zeros(2*num_samples, dtype=[('input',  float, 2),
                                                 ('output', float, 1)])
        samples['input'] = np.vstack([pos_samples, neg_samples])
        samples['output'] = np.hstack([np.ones((num_samples,)),
                                      -np.ones((num_samples,))])
        return samples

    samples = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])

    # Example 1 : OR logical function
    # -------------------------------------------------------------------------
    print("Learning the OR logical function")
    samples[0] = (0,0), -1
    samples[1] = (1,0), +1
    samples[2] = (0,1), +1
    samples[3] = (1,1), +1
    network = train(samples)
    test(network, samples)

    # Example 2 : AND logical function
    # -------------------------------------------------------------------------
    print("Learning the AND logical function")
    samples[0] = (0,0), -1
    samples[1] = (1,0), -1
    samples[2] = (0,1), -1
    samples[3] = (1,1), +1
    network = train(samples)
    test(network, samples)

    # Example 3 : XOR logical function
    # -------------------------------------------------------------------------
    print("Failed at learning the XOR logical function")
    samples[0] = (0,0), -1
    samples[1] = (1,0), +1
    samples[2] = (0,1), +1
    samples[3] = (1,1), -1
    network = train(samples)
    test(network, samples)

    # Example 4 : Blobs
    # -------------------------------------------------------------------------
    print("Learning the blobs dataset")
    train_samples = make_blobs(50)
    test_samples  = make_blobs(500)

    network = train(train_samples)
    empirical_accuracy = accuracy(network, train_samples)
    real_accuracy = accuracy(network, test_samples)

    print("Empirical accuracy : {} %".format(100 * empirical_accuracy))
    print("Estimated real accuracy : {} %".format(100 * real_accuracy))

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
