# -----------------------------------------------------------------------------
# Copyright 2019 (C) Jeremy Fix
# Released under a BSD two-clauses license
#
# References: Y. Freund, R. E. Schapire. "Large margin classification using
#             the perceptron algorithm". In: 11th Annual Conference on
#             Computational Learning Theory, New York, NY, 209-217, 1998.
#             DOI:10.1023/A:1007662407062
# -----------------------------------------------------------------------------
import numpy as np

import perceptron

# WIP : Online algorithm of Fig. 1
# We should get rid of the dependance to perceptron just by
#  hosting our own weights .. which implies a growing 3D numpy array of weigths
# That way we can vectorize and make the algorithm much faster

class VotedPerceptron:

    def __init__(self, n, m):
        ''' Initialization of the voted perceptron with given sizes.  '''

        self.n = n
        self.m = m
        self.reset()

    def reset(self):
        ''' Cleans up the learned perceptrons '''
        self.perceptrons = [perceptron.Perceptron(self.n, self.m)]
        self.perceptrons[-1].reset()
        self.votes = [0]

    def learn(self, input_sample, output_sample, lrate):
        ''' The learning function : a single sample is expected '''
        o = self.perceptrons[-1].propagate_forward(input_sample)
        if o == output_sample:
            self.votes[-1] += 1
        else:
            # Create a new perceptron
            self.perceptrons.append(perceptron.Perceptron(self.n, self.m))
            self.votes.append(0)

    def __call__(self, input_sample):
        ''' Prediction step '''
        s = 0
        for ck, pk in zip(self.votes, self.perceptrons):
            s += ck * (2 * pk.propagate_forward(input_sample) - 1)
        return s > 0

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def learn(network,samples, epochs=250, lrate=.1, momentum=0.1):
        # Train
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.learn(samples['input'][n], samples['output'][n], lrate)
        # Test
        for i in range(samples.size):
            o = network(samples['input'][i])
            print(i, samples['input'][i], '%.2f' % o[0],
                  '(expected %.2f)' % samples['output'][i])


    network = VotedPerceptron(2,1)
    samples = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])

    # Example 1 : OR logical function
    # -------------------------------------------------------------------------
    print("Learning the OR logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 1
    samples[2] = (0,1), 1
    samples[3] = (1,1), 1
    learn(network, samples)

    # Example 2 : AND logical function
    # -------------------------------------------------------------------------
    print("Learning the AND logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 0
    samples[2] = (0,1), 0
    samples[3] = (1,1), 1
    learn(network, samples)

    # Example 3 : XOR logical function
    # -------------------------------------------------------------------------
    print("Failed at learning the XOR logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 1
    samples[2] = (0,1), 1
    samples[3] = (1,1), 0
    learn(network, samples)

    # Example 4 : Large dimensional space
    # -------------------------------------------------------------------------
    print("Everything is linearly separable in large dimensional space :)")
