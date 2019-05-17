# -----------------------------------------------------------------------------
# Copyright 2019 (C) Jeremy Fix
# Released under a BSD two-clauses license
#
# References: Aizerman, M. A., Braverman, E. A. and Rozonoer, L.. " Theoretical
#             foundations of the potential function method in pattern
#             recognition learning.." Paper presented at the meeting of the
#             Automation and Remote Control,, 1964.
# -----------------------------------------------------------------------------

import numpy as np

class KernelPerceptron:

    def __init__(self, n, m):
        ''' Initialization of the perceptron with given sizes.  '''

        self.input  = np.ones(n+1)
        self.output = np.ones(m)

    def __call__(self, inputs):
        pass
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Generate the moon dataset

    # Learn

    # Plot
