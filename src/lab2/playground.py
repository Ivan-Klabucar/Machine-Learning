import random, math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import data
from SVM import SVM
from kernals import *


inputs, targets, classA, classB = data.load_data()

model = SVM(get_RBF_kernal(1.))
model.train(inputs, targets)

# Commented sections are only for when you use C argument in SVM
# noisy_supp_vecs = []
# for i, a in enumerate(model.supp_alfai):
#     if not a < (C_val - 1e-5):
#         noisy_supp_vecs.append(model.supp_vecs[i])


plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.plot([p[0] for p in model.supp_vecs], [p[1] for p in model.supp_vecs], 'g+')
#plt.plot([p[0] for p in noisy_supp_vecs], [p[1] for p in noisy_supp_vecs], marker='o', markerfacecolor='none', markeredgecolor='m', linestyle='None')
plt.axis('equal') # Force same s c a l e on both axes
plt.savefig('svmplot.pdf') # Save a copy in a f i l e

xgrid=np.linspace(-5, 5)
ygrid=np.linspace(-4, 4)
grid=np.array([[model.ind(np.array([x, y])) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors =('red', 'black', 'blue'), linewidths=(1, 3, 1))



plt.show() # Show the p l o t on the screen