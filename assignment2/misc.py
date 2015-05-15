##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    e = (math.sqrt(6)/math.sqrt(m+n))
    A0 = (random.rand(m,n)*2.*e) - e

    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0
