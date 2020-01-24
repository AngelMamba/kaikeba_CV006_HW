"""
Target: linear regression with multiple variables to predict the prices of houses
File 'house_price_training_data.txt' contains a training set of housing prices in Portland, Oregon.
The first column is the size of the house (in square feet).
The second column is the number of bedrooms.
The third column is the price of the house.
Reference: machine learning course exercises by Andrew Ng from Coursera
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def featureNormalize(x):
    x_mu = np.mean(x, axis=0)  # mean
    x_mu = x_mu.reshape((1, x.shape[1]))
    x_std = np.std(x, axis=0)  # standard deviation
    x_std = x_std.reshape((1, x.shape[1]))
    x_normalize = (X - x_mu) / x_std
    return x_normalize, x_mu, x_std


def computeCost(X, y, w):
    m = y.shape[0]
    J = np.sum((X @ w - y) ** 2) / (2 * m)
    return J


def gradientDescent(X, y, w, alpha, n_iter):
    m = y.shape[0]
    J_history = np.zeros((n_iter, 1))

    for i in range(n_iter):
        w -= alpha / m * (X.T @ (X @ w - y))
        J_history[i, 0] = computeCost(X, y, w)

    return w, J_history


#  Load sample data
sample = np.loadtxt('house_price_training_data.txt', delimiter=',')
X = sample[:, 0:2]  # features
y = sample[:, 2]  # real value
y = y[:, None]
m = sample.shape[0]  # number of training samples

# Feature normalization
X_normalize, X_mu, X_std = featureNormalize(X)
X_0 = np.ones((X.shape[0], 1))  # define intercept term
X_ready = np.append(X_0, X_normalize, axis=1)  # add intercept term to X, which can be considered as 'b'

# Initialization
w = np.random.random_sample((X_ready.shape[1], 1))  # initialize weight
alpha = 0.01  # initialize learning rate
n_iter = 500  # initialize iterations

# Update w based on gradient descent
w, J_history = gradientDescent(X_ready, y, w, alpha, n_iter)

# Predict value
predict = X_ready @ w

# Plot
plt.figure(1)
plt.plot(range(n_iter), J_history)
plt.title('Cost function')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.grid(True)

fig = plt.figure(2)
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], y, 'bo', label='Actual')
ax.scatter(X[:, 0], X[:, 1], predict, 'r*', label='Predict')
ax.legend(loc='best')
ax.set_xlabel('Size of the house')
ax.set_ylabel('Number of bedrooms')
ax.set_zlabel('Price of the house')

plt.show()


