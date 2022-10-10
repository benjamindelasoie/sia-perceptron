import random

import numpy as np

from normalize import escale_all

rng = np.random.default_rng()


def h(X, weights):
    return np.dot(X, weights)


def step_function(x):
    return np.where(x >= 0, 1., -1.)


def id_function(x):
    return x


def one_function(x):
    return 1.


def tanh(x):
    beta = 0.1
    return np.tanh(beta * x)


def tanh_derivative(x):
    beta = 0.1
    return beta * (1 - (np.tanh(beta * x)) ** 2)


def logistic_function(x):
    beta = 10
    return 1 / (1 + np.exp(-2 * beta * x))


def logistic_derivative(x):
    beta = 10
    return 2 * beta * logistic_function(x) * (1 - logistic_function(x))


def simple_error(X, y, w, f):
    hs = h(X, w)
    os = f(hs)
    error = np.sum(np.abs(y - os))
    return error


def mean_squared_error(X, y, w, f):
    hs = h(X, w)
    os = f(hs)
    return np.mean((y - os) ** 2)


def delta_w_nonsimple(lr, X, mu, y, w, g, g_prime):
    hs = h(X, w)
    # print("hs",  hs)
    os = g(hs)
    g_primes = np.apply_along_axis(g_prime, 0, hs)

    # print("gprime", g_primes)

    X = X.T

    delta = np.zeros(w.T.shape)

    for i in range(w.T.size):
        delta[0][i] = lr * 2 * np.mean((y - os) * g_primes * X[i].reshape(os.shape))

    return delta


def delta_w_simple(lr, X, mu, y, w, g, g_prime):
    exc = h(X[mu], w)
    o = g(exc)
    return lr * (y[mu] - o) * X[mu]


def add_bias(X, value):
    n, *p = X.shape
    bias = np.repeat(value, n).reshape((n, 1))
    n = np.concatenate((bias, X), axis=1)
    return n


class Perceptron:
    def __init__(self, learning_rate, epochs, error_min) -> None:
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.g = step_function
        self.error_function = simple_error
        self.bias = -1
        self.delta_w_f = delta_w_simple
        self.g_prime = one_function  ## para que no tire error
        self.error_min = error_min

    def train(self, X, y):
        X = add_bias(X, self.bias)
        n, p = X.shape
        weights = np.zeros(shape=(p, 1))
        mus = np.array(range(n))

        i = 0
        error = 0
        error_min = self.error_min
        w_min = weights

        while (error_min > 0.005 and i < self.epochs):
            random.shuffle(mus)
            # take a random sample
            for mu in mus:

                # mu = rng.integers(0, n)
                ##exc = h(X[mu], weights)          # excitement
                ##o = self.g(exc)              # activation

                # update weights
                delta_w = self.delta_w_f(self.lr, X, mu, y, weights, self.g, self.g_prime)
                delta_w = delta_w.reshape(p, 1)

                weights = weights + delta_w

                # calculate error
                error = self.error_function(X, y, weights, self.g)
                if (error < error_min):
                    error_min = error
                    w_min = weights

            i = i + 1

        self.weights = w_min
        error = self.error_function(X, y, weights, self.g)  ##podria ser error_min
        return self.weights, error, i

    def predict(self, X):
        if self.weights is not None:
            X = add_bias(X, self.bias)
            hs = h(X, self.weights)
            predictions = np.apply_along_axis(self.g, 1, hs)
            return predictions


class LinearPerceptron(Perceptron):
    def __init__(self, learning_rate, epochs, error_min) -> None:
        super().__init__(learning_rate, epochs, error_min)
        self.g = id_function
        self.g_prime = one_function
        self.error_function = mean_squared_error
        self.bias = 1
        self.delta_w_f = delta_w_nonsimple
        self.error_min = error_min


class NonLinearPerceptron(Perceptron):
    def __init__(self, learning_rate, epochs, error_min) -> None:
        super().__init__(learning_rate, epochs, error_min)
        self.g = tanh
        self.g_prime = tanh_derivative
        self.error_function = mean_squared_error
        self.bias = 1
        self.delta_w_f = delta_w_nonsimple
        self.error_min = error_min