import random, math
import numpy as np


def load_data():
    classA = np.concatenate(
    (np.random.randn(10 , 2) * 0.25 + [1.5, 0.5], np.random.randn(10, 2) * 0.25 + [-1.5, 0.5]))
    classB = np.random.randn(20, 2) * 0.3 + [0.0, -0.5]
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
    N = inputs.shape[0] # Number of rows ( samples )
    permute= list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    return inputs, targets, classA, classB
