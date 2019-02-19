# -----------------------------------------------------------------------------
# Copyright 2019 (C) Nicolas P. Rougier
# Released under a BSD two-clauses license
#
# Reference: Grossberg, S. (1987). Competitive learning: From interactive
#            activation to adaptive resonance, Cognitive Science, 11, 23-63.
# -----------------------------------------------------------------------------
import numpy as np


class ART:
    ''' ART class '''

    def __init__(self, n=5, m=10, rho=.5):
        '''
        Create network with specified shape

        Parameters:
        -----------
        n : int
            Size of input
        m : int
            Maximum number of internal units 
        rho : float
            Vigilance parameter
        '''
        # Comparison layer
        self.F1 = np.ones(n)
        # Recognition layer
        self.F2 = np.ones(m)
        # Feed-forward weights
        self.Wf = np.random.random((m,n))
        # Feed-back weights
        self.Wb = np.random.random((n,m))
        # Vigilance
        self.rho = rho
        # Number of active units in F2
        self.active = 0


    def learn(self, X):
        ''' Learn X '''

        # Compute F2 output and sort them (I)
        self.F2[...] = np.dot(self.Wf, X)
        I = np.argsort(self.F2[:self.active].ravel())[::-1]

        for i in I:
            # Check if nearest memory is above the vigilance level
            d = (self.Wb[:,i]*X).sum()/X.sum()
            if d >= self.rho:
                # Learn data
                self.Wb[:,i] *= X
                self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
                return self.Wb[:,i], i

        # No match found, increase the number of active units
        # and make the newly active unit to learn data
        if self.active < self.F2.size:
            i = self.active
            self.Wb[:,i] *= X
            self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
            self.active += 1
            return self.Wb[:,i], i

        return None,None


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    np.random.seed(1)

    # Example 1 : very simple data
    # -------------------------------------------------------------------------
    network = ART( 5, 10, rho=0.5)
    data = ["   O ",
            "  O O",
            "    O",
            "  O O",
            "    O",
            "  O O",
            "    O",
            " OO O",
            " OO  ",
            " OO O",
            " OO  ",
            "OOO  ",
            "OO   ",
            "O    ",
            "OO   ",
            "OOO  ",
            "OOOO ",
            "OOOOO",
            "O    ",
            " O   ",
            "  O  ",
            "   O ",
            "    O",
            "  O O",
            " OO O",
            " OO  ",
            "OOO  ",
            "OO   ",
            "OOOO ",
            "OOOOO"]
    X = np.zeros(len(data[0]))
    for i in range(len(data)):
        for j in range(len(data[i])):
            X[j] = (data[i][j] == 'O')
        Z, k = network.learn(X)
        print("|%s|"%data[i],"-> class", k)

   
    
    # Example 2 : Learning letters
    # -------------------------------------------------------------------------
    def letter_to_array(letter):
        ''' Convert a letter to a numpy array '''
        shape = len(letter), len(letter[0])
        Z = np.zeros(shape, dtype=int)
        for row in range(Z.shape[0]):
            for column in range(Z.shape[1]):
                if letter[row][column] == '#':
                    Z[row][column] = 1
        return Z

    def print_letter(Z):
        ''' Print an array as if it was a letter'''
        for row in range(Z.shape[0]):
            for col in range(Z.shape[1]):
                if Z[row,col]:
                    print( '#', end="" )
                else:
                    print( ' ', end="" )
            print( )
       
    A = letter_to_array( [' #### ',
                          '#    #',
                          '#    #',
                          '######',
                          '#    #',
                          '#    #',
                          '#    #'] )
    B = letter_to_array( ['##### ',
                          '#    #',
                          '#    #',
                          '##### ',
                          '#    #',
                          '#    #',
                          '##### '] )
    C = letter_to_array( [' #### ',
                          '#    #',
                          '#     ',
                          '#     ',
                          '#     ',
                          '#    #',
                          ' #### '] )
    D = letter_to_array( ['##### ',
                          '#    #',
                          '#    #',
                          '#    #',
                          '#    #',
                          '#    #',
                          '##### '] )
    E = letter_to_array( ['######',
                          '#     ',
                          '#     ',
                          '####  ',
                          '#     ',
                          '#     ',
                          '######'] )
    F = letter_to_array( ['######',
                          '#     ',
                          '#     ',
                          '####  ',
                          '#     ',
                          '#     ',
                          '#     '] )

    samples = [A,B,C,D,E,F]
    network = ART( 6*7, 10, rho=0.15 )

    for i in range(len(samples)):
        Z, k = network.learn(samples[i].ravel())
        print("%c"%(ord('A')+i),"-> class",k)
        print_letter(Z.reshape(7,6))
