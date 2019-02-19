# -----------------------------------------------------------------------------
# Copyright 2019 (C) Nicolas P. Rougier
# Released under a BSD two-clauses license
#
# References: Rumelhart, David E., Geoffrey E. Hinton, and R. J. Williams.
#             "Learning Internal Representations by Error Propagation". David
#             E. Rumelhart, James L. McClelland, and the PDP research
#             group. (editors), Parallel distributed processing: Explorations
#             in the microstructure of cognition, Volume 1: Foundation. MIT
#             Press, 1986.
# -----------------------------------------------------------------------------
import numpy as np

def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2

class MLP:
    ''' Multi-layer perceptron class. '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0,]*len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],
                                                 self.weights[i-1]))

        # Return output
        return self.layers[-1]


    def propagate_backward(self, target, lrate=0.05, momentum=0.5):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)
            
        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error**2).sum()


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    def learn(network,samples, epochs=2500, lrate=.1, momentum=0.1):
        # Train 
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            network.propagate_backward( samples['output'][n], lrate, momentum )
        # Test
        for i in range(samples.size):
            o = network.propagate_forward( samples['input'][i] )
            print (i, samples['input'][i], '%.2f' % o[0],)
            print ('(expected %.2f)' % samples['output'][i])
        print()

    network = MLP(2,2,1)
    samples = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])

    # Example 1 : OR logical function
    # -------------------------------------------------------------------------
    print( "Learning the OR logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 1
    samples[2] = (0,1), 1
    samples[3] = (1,1), 1
    learn(network, samples)

    # Example 2 : AND logical function
    # -------------------------------------------------------------------------
    print( "Learning the AND logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 0
    samples[2] = (0,1), 0
    samples[3] = (1,1), 1
    learn(network, samples)

    # Example 3 : XOR logical function
    # -------------------------------------------------------------------------
    print( "Learning the XOR logical function")
    network.reset()
    samples[0] = (0,0), 0
    samples[1] = (1,0), 1
    samples[2] = (0,1), 1
    samples[3] = (1,1), 0
    learn(network, samples)

    # Example 4 : Learning sin(x)
    # -------------------------------------------------------------------------
    print ("Learning the sin function")
    network = MLP(1,20,1)
    n = 1000
    samples = np.zeros(n, dtype=[('x',  float, 1), ('y', float, 1)])
    samples['x'] = np.linspace(0,1,n)
    samples['y'] = np.sin(samples['x']*np.pi*2)*0.5

    for i in range(50000):
        n = np.random.randint(samples.size)
        network.propagate_forward(samples['x'][n])
        network.propagate_backward(samples['y'][n], lrate=0.01, momentum=0.1)

    plt.figure(figsize=(10,5))
    # Draw real function
    x,y = samples['x'],samples['y']
    plt.plot(x,y,color='b',lw=1)
    # Draw network approximated function
    for i in range(samples.shape[0]):
        y[i] = network.propagate_forward(x[i])
    plt.plot(x,y,color='r',lw=3)
    plt.show()
