import numpy as np

def linear_kernal(x, y):
    return np.dot(x, y)

def get_polynomial_kernel(p):
    def poly_kernal(x, y):
        return (np.dot(x,y) + 1)**p
    return poly_kernal

def get_RBF_kernal(sigma):
    def RBF_kernal(x, y):
        return np.exp(-(np.linalg.norm(x - y)**2)/(2*sigma**2))
    return RBF_kernal
