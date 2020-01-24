"""
Target: build a classification model that estimates an applicant's
probability of admission based the scores from those two exams.
File 'score.txt' contains the exam scores and the corresponding result admission
The first two columns are the exam scores.
The third column is the result.
Reference: machine learning course exercises by Andrew Ng from Coursera
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def costFunction(w, X, y):
    w = np.array(w).reshape((np.size(w), 1))
    m = np.size(y)
    h = sigmoid(np.dot(X, w))
    J = 1/m*(-np.dot(y.T, np.log(h)) - np.dot((1-y.T), np.log(1-h)))
    return J.flatten()


def gradient(w, X, y):
    w = np.array(w).reshape((np.size(w), 1))
    m = np.size(y)
    h = sigmoid(np.dot(X, w))
    grad = 1/m*np.dot(X.T, h-y)
    return grad.flatten()


#  Load sample data
sample = np.loadtxt('score.txt', delimiter=',')
X = sample[:, 0:2]
y = sample[:, 2]
y = y[:, None]
m = sample.shape[0]  # number of training samples


# Sample prepare
X_0 = np.ones((X.shape[0], 1))  # define intercept term
X_ready = np.append(X_0, X, axis=1)  # add intercept term to X, which can be considered as 'b'

# Initialize w and b as in y = wx + b
w0 = np.zeros((X_ready.shape[1], 1))

# Compute optimized result
optimizeResult = op.minimize(fun=costFunction, x0=w0, args=(X_ready, y), method='TNC', jac=gradient)
w = optimizeResult.x
cost = optimizeResult.fun
print('w = ', w)
print('cost = ', cost)

plt.figure()
pos = np.array(np.nonzero(y > 0.5))
neg = np.array(np.nonzero(y < 0.5))
plt.plot(X[pos[0, :], 0], X[pos[0, :], 1], 'ro')  # plot positive results
plt.plot(X[neg[0, :], 0], X[neg[0, :], 1], 'b*')  # plot negative results

plot_x = np.array([np.min(X_ready[:, 1]), np.max(X_ready[:, 2])])
plot_y = (-1/w[2])*(w[1]*plot_x+w[0])
plt.plot(plot_x, plot_y)  # plot classification

plt.legend(labels=['Admitted', 'Not admitted'])

plt.show()
