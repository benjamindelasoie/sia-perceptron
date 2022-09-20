from math import exp
import numpy as np

rng = np.random.default_rng()

def step_function(z):
  return np.where(z > 0, 1., -1.)

def id_function(z):
  return z

def one_function(x):
  return 1.

def logistic_function(x):
  return (1 / (1 + exp(-x)))

def logistic_derivative(x):
  f = logistic_function
  return f(x) * (1 - f(x))

def simple_error(X, y, w, f):
  hs = np.dot(X, w)
  os = f(hs)
  error = np.sum(np.abs(y - os))
  return error

def mean_squared_error(X, y, w, f):
  hs = np.dot(X, w)
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
    self.g_prime = one_function
    self.error_function = simple_error

  def train(self, X, y):
    X = add_bias(X)
    n, p = X.shape
    i = 0
    weights = np.empty(shape=(p, 1))
    error = 0
    error_min = 100000000
    w_min = weights
    
    while (error_min > 0 and i < self.epochs):
      mu = rng.integers(0, n)    # pick a random data point
      print(f"mu : {mu}")
      print("X[mu]")
      print(X[mu])
      print("weights")
      print(weights )
      h = np.dot(X[mu], weights) # excitement 
      print(f"h(X[mu]) : {h}")
      o = self.g(h)              # activation
      print(f"o(h) : {o}")
      o_prime = self.g_prime(h)  # its derivate
      print(f"expected : {y[mu]}")

      # update weights if necessary
      delta_w = self.lr * (y[mu] - o) * o_prime * X[mu]
      print("delta_w shape:")
      delta_w = delta_w.reshape(p, 1)
      print(delta_w.shape)

      print("delta_w")
      print(delta_w)

      weights = weights + delta_w

      # should calculate error here
      error = self.error_function(X, y, weights, self.g)
      if (error < error_min):
        error_min = error
        w_min = weights
      
      print(f"iter: {i}")
      print("weights")
      print(weights)
      print(f"error = {error}")
      i = i + 1

    self.weights = w_min
    error = self.error_function(X, y, weights, self.g)
    return error

  def predict(self, x_hat):
    if self.weights is not None:
      x_hat = add_bias(x_hat)
      excitements = np.dot(x_hat, self.weights)
      predictions = np.apply_along_axis(self.o, 1, excitements)
      return predictions

# NO ESTAN FUNCIONANDO

class LinearPerceptron(Perceptron):
  def __init__(self, learning_rate, epochs) -> None:
    super().__init__(learning_rate, epochs)
    self.g = id_function
    self.g_prime = one_function
    self.error_function = mean_squared_error

class NonLinearPerceptron(Perceptron):
  def __init__(self, learning_rate, epochs) -> None:
    super().__init__(learning_rate, epochs)
    self.g = logistic_function
    self.g_prime = logistic_derivative
    self.error = mean_squared_error 