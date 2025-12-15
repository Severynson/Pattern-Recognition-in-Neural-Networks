import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(z):
    # stable softmax
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)