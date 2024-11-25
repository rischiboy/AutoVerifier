import numpy as np

def spu(x):
    if x > 0:
        return x**2 - 0.5
    else:
        return sigmoid(-x) -1

def spu_deriv(x):
    if x <= 0:
        return -1 * sigmoid(-x) * (1 - sigmoid(-x))
    # if x == 0:
        # return 0
    else:
        return 2*x

def sigmoid(x):
    return 1/(1 + np.exp(-x))