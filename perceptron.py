from math import exp
import numpy as np

rng = np.random.default_rng()

def h(X, weights):
  return np.dot(X, weights)

def step_function(x):
  return np.where(x >= 0, 1., -1.)

def id_function(x):
  return x

def one_function(x):
  return 1.

def logistic_function(x):
  return (1 / (1 + np.exp(-x)))

def logistic_derivative(x):
  f = logistic_function
  return f(x) * (1 - f(x))

def simple_error(X, y, w, f):
  hs = h(X, w)
  os = f(hs)
  error = np.sum(np.abs(y - os))
  return error

def mean_squared_error(X, y, w, f):
  hs = h(X, w)
  os = f(hs)
  error = 0.5 * np.sum((y - os) ** 2)
  return error

def add_bias(X):
  n, *p = X.shape
  bias = np.repeat(-1, n).reshape((n, 1))
  n = np.concatenate((bias, X), axis=1)
  return n

class Perceptron:
  def __init__(self, learning_rate, epochs) -> None:
    self.lr = learning_rate
    self.epochs = epochs
    self.weights = None
    self.g = step_function
    self.error_function = simple_error

  def train(self, X, y):
    X = add_bias(X)
    n, p = X.shape
    weights = np.zeros(shape=(p, 1))

    i = 0
    error = 0
    error_min = 100000000
    w_min = weights
    
    while (error_min > 0 and i < self.epochs):
      # take a random sample
      mu = rng.integers(0, n)
      exc = h(X[mu], weights)          # excitement 
      o = self.g(exc)              # activation

      # update weights
      delta_w = self.lr * (y[mu] - o) * X[mu]
      delta_w = delta_w.reshape(p, 1)

      weights = weights + delta_w

      # calculate error
      error = self.error_function(X, y, weights, self.g)
      if (error < error_min):
        error_min = error
        w_min = weights
      
      i = i + 1

    self.weights = w_min
    error = self.error_function(X, y, weights, self.g)
    return self.weights, error, i

  def predict(self, X):
    if self.weights is not None:
      X = add_bias(X)
      hs = h(X, self.weights)
      predictions = np.apply_along_axis(self.g, 1, hs)
      return predictions

# NO ESTAN FUNCIONANDO

class LinearPerceptron(Perceptron):
  def __init__(self, learning_rate, epochs) -> None:
    super().__init__(learning_rate, epochs)
    self.g = id_function
    self.g_prime = one_function
    self.error_function = mean_squared_error

  def train(self, X, y):
    # X = add_bias(X)
    n, p = X.shape
    i = 0
    weights = np.zeros(shape=(p, 1))
    error = 0
    error_min = 100000000
    w_min = weights
    
    while (error_min > 0 and i < self.epochs):
      mu = rng.integers(0, n)    # pick a random data point
      exc = h(X, weights)        # excitement 
      o = self.g(exc)              # activation
      o_prime = self.g_prime(exc)  # its derivate

      os = X @ weights

      # update weights if necessary
      delta_w = self.lr * np.sum((y - os) * X, axis=0)
      delta_w = delta_w.reshape(p, 1)


      weights = weights + delta_w

      # should calculate error here
      error = self.error_function(X, y, weights, self.g)
      if (error < error_min):
        error_min = error
        w_min = weights
      
      i = i + 1

    self.weights = w_min
    error = self.error_function(X, y, weights, self.g)
    return self.weights, error

class NonLinearPerceptron(Perceptron):
  def __init__(self, learning_rate, epochs) -> None:
    super().__init__(learning_rate, epochs)
    self.g = logistic_function
    self.g_prime = logistic_derivative
    self.error_function = mean_squared_error 