import random

import numpy as np

from normalize import escale_all

rng = np.random.default_rng()

def h(X, weights):
  return np.dot(X, weights)

def step_function(x, beta):
  return np.where(x >= 0, 1., -1.)

def id_function(x, beta):
  return x

def one_function(x, beta):
  return 1.

def tanh(x, beta):
  #beta = 0.1
  return np.tanh(beta * x)


def tanh_derivative(x, beta):
  #beta = 0.1
  return beta * (1 - (tanh(beta * x))**2)

def logistic_function(x):
  beta = 10
  return (1 / (1 + np.exp(-2 * beta * x)))


def logistic_derivative(x):
  beta = 10
  return 2 * beta * logistic_function(x) * (1 - logistic_function(x))

def simple_error(X, y, w, g, beta):
  hs = h(X, w)
  os = g(hs, beta)
  error = np.sum(np.abs(y - os))
  return error


def mean_squared_error(X, y, w, g, beta):
  hs = h(X, w)
  os = g(hs, beta)
  return np.mean((y - os) ** 2)

def delta_w_nonsimple(lr, X, mu, y, w, g, g_prime, beta):
    hs = h(X, w)
    #print("hs",  hs)
    os = g(hs, beta)
    g_primes = np.apply_along_axis(g_prime, 0, hs, [beta])

    #print("gprime", g_primes)

    X = X.T

    delta = np.zeros(w.T.shape)


    for i in range(w.T.size):
      delta[0][i] = lr * 2 * np.mean((y - os) * g_primes * X[i].reshape(os.shape))

    return delta


##NO SE USA
#def delta_w_linear(lr, X, mu, y, w, g, g_prime):
  #os = g(h(X, w))
  #for i in range(y.size):
  #  for j in range()
  #X = X.T
  #y=y.T
  #os=os.T

  #delta = np.zeros(w.T.shape)


  #for i in range(w.T.size):
    #print(lr * 2 * np.mean((y - os) * X[i]))
    #delta[0][i] = lr * 2 * np.mean((y - os) * X[i].reshape(os.shape))


  #print("delta", delta)
  #r =np.array(lr * 2 * np.mean(y - os) * X)
  #print("r", r)
  #print(np.sum(r, axis=0))
  #return delta


def delta_w_simple(lr, X, mu, y, w, g, g_prime, beta):
  exc = h(X[mu], w)
  o = g(exc, beta)
  return lr * (y[mu] - o) * X[mu]

def add_bias(X, value):
  n, *p = X.shape
  bias = np.repeat(value, n).reshape((n, 1))
  n = np.concatenate((bias, X), axis=1)
  return n

class Perceptron:
  def __init__(self, learning_rate, epochs, beta) -> None:
    self.lr = learning_rate
    self.epochs = epochs
    self.weights = None
    self.g = step_function
    self.error_function = simple_error
    self.bias = -1
    self.beta = beta
    self.delta_w_f = delta_w_simple
    self.g_prime = one_function ## para que no tire error

  def train(self, X, y):
    X = add_bias(X, self.bias)
    n, p = X.shape
    weights = np.zeros(shape=(p, 1))
    mus = np.array(range(n))
    errors = []
    epochs = []


    i = 0
    error = 0
    error_min = 100000000
    w_min = weights
    
    while (error_min > 0.005 and i < self.epochs):
      random.shuffle(mus)
      # take a random sample
      for mu in mus:

      #mu = rng.integers(0, n)
      ##exc = h(X[mu], weights)          # excitement
      ##o = self.g(exc)              # activation

      # update weights
        delta_w = self.delta_w_f(self.lr, X, mu, y, weights, self.g, self.g_prime, self.beta)
        delta_w = delta_w.reshape(p, 1)

        weights = weights + delta_w

      # calculate error
        error = self.error_function(X, y, weights, self.g, self.beta)
        #errors.append(error)
        if (error < error_min):
          error_min = error
          w_min = weights

      errors.append(error)
      epochs.append(i)
      i = i + 1

    self.weights = w_min
    #error = self.error_function(X, y, weights, self.g) ##podria ser error_min
    return self.weights, errors, epochs

  def predict(self, X):
    if self.weights is not None:
      X = add_bias(X, self.bias)
      hs = h(X, self.weights)
      predictions = np.apply_along_axis(self.g, 1, hs, [self.beta])
      return predictions


class LinearPerceptron(Perceptron):
  def __init__(self, learning_rate, epochs, beta) -> None:
    super().__init__(learning_rate, epochs, beta)
    self.g = id_function
    self.g_prime = one_function
    self.error_function = mean_squared_error
    self.bias = 1
    self.delta_w_f = delta_w_nonsimple


class NonLinearPerceptron(Perceptron):
  def __init__(self, learning_rate, epochs, beta) -> None:
    super().__init__(learning_rate, epochs, beta)
    self.g = tanh
    self.g_prime = tanh_derivative
    self.error_function = mean_squared_error
    self.bias = 1
    self.delta_w_f = delta_w_nonsimple