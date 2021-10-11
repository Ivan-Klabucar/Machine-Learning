import random, math
import numpy as np
from scipy.optimize import minimize


class SVM:
    def __init__(self, kernal_func, C=None):
        self.kernal_func = kernal_func
        self.P = None     # Precomputed matrix of ti * tj * K(xi, xj)
        self.C = C
        self.supp_vecs = []
        self.supp_vecs_alfai_ti = []
        self.b = None
    
    def ind(self, x):
        res = 0
        for i, sv in enumerate(self.supp_vecs):
            res += self.supp_vecs_alfai_ti[i] * self.kernal_func(x, sv)
        return res - self.b

    def objective_func(self, alfa):
        alfa_vec = np.reshape(alfa, (-1,1))
        alfa_ij_matrix = alfa_vec @ alfa_vec.T
        return 0.5 * np.sum(np.multiply(alfa_ij_matrix, self.P)) - np.sum(alfa_vec)

    def precompute(self, training_set, training_labels):
        t_labels = np.reshape(training_labels, (-1, 1))
        self.P = t_labels @ t_labels.T
        for i, xi in enumerate(training_set):
            for j, xj in enumerate(training_set):
                self.P[i][j] *= self.kernal_func(xi, xj)
    
    def zerofun(self, alfa, training_labels):
        return np.dot(alfa, training_labels)

    def train(self, training_set, training_labels):
        N = training_set.shape[0]
        self.precompute(training_set, training_labels)
        ret = minimize( \
            self.objective_func, \
            np.zeros(N), \
            bounds=[(0, self.C) for b in range(N)], \
            constraints={'type':'eq', 'fun': lambda a: self.zerofun(a, training_labels)} \
            )
        if not ret['success']: print('Not linearly separable, optimization failed.')
        alpha = ret['x']
        svec_for_b_calculation = None
        svec_for_b_t = None
        for i, a in enumerate(alpha):
            if a > 1e-5:
                self.supp_vecs.append(training_set[i])
                self.supp_vecs_alfai_ti.append(training_labels[i] * a)
                if not self.C or a <= (self.C - 1e-5): # not sure if we need to allow for tolerance
                    svec_for_b_calculation = training_set[i]
                    svec_for_b_t = training_labels[i]

        self.b = 0
        for i, sv in enumerate(self.supp_vecs):
            self.b += self.supp_vecs_alfai_ti[i] * self.kernal_func(svec_for_b_calculation, sv)
        self.b -= svec_for_b_t




