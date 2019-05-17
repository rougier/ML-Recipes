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

class VotedPerceptron:

    def __init__(self, n, m):
        ''' Initialization of the voted perceptron with given sizes.  '''
        self.n = n
        self.m = m
        self.input  = np.ones(n+1)
        self.output = np.ones(m)
        self.reset()

    def reset(self):
        ''' Cleans up the learned perceptrons '''
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

    def learn(network, samples, epochs=250):
        # Train
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.learn(samples['input'][n], samples['output'][n])
        # Test
        for i in range(samples.size):
            o = network(samples['input'][i])
            print(i, samples['input'][i], '%.2f' % o,
                  '(expected %.2f)' % samples['output'][i])


    network = VotedPerceptron(2,1)
    samples = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])

    # Example 1 : OR logical function
    # -------------------------------------------------------------------------
    print("Learning the OR logical function")
    network.reset()
    samples[0] = (0,0), -1
    samples[1] = (1,0), +1
    samples[2] = (0,1), +1
    samples[3] = (1,1), +1
    learn(network, samples)

    # Example 2 : AND logical function
    # -------------------------------------------------------------------------
    print("Learning the AND logical function")
    network.reset()
    samples[0] = (0,0), -1
    samples[1] = (1,0), -1
    samples[2] = (0,1), -1
    samples[3] = (1,1), +1
    learn(network, samples)

    # Example 3 : XOR logical function
    # -------------------------------------------------------------------------
    print("Failed at learning the XOR logical function")
    network.reset()
    samples[0] = (0,0), -1
    samples[1] = (1,0), +1
    samples[2] = (0,1), +1
    samples[3] = (1,1), -1
    learn(network, samples)
