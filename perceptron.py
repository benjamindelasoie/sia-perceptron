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
  beta = 10
  return np.tanh(beta * x)


def tanh_derivative(x):
  beta = 10
  return beta * (1 - tanh(x))

def logistic_function(x):
  beta = 10
  return (1 / (1 + np.exp(-2 * beta * x)))


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
  error = ((y - os) ** 2).mean()
  return error

def delta_w_nonlinear(lr, X, mu, y, w, g, g_prime):
    os = g(h(X, w))
    hmu = h(X[mu], w)
    return lr * 2 * np.mean(y - os) * g_prime(hmu) * X[mu]

def delta_w_linear(lr, X, mu, y, w, g, g_prime):
  os = g(h(X, w))
  return lr * 2 * np.mean(y - os) * X[mu]


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
  def __init__(self, learning_rate, epochs) -> None:
    self.lr = learning_rate
    self.epochs = epochs
    self.weights = None
    self.g = step_function
    self.error_function = simple_error
    self.bias = -1
    self.delta_w_f= delta_w_simple
    self.g_prime = one_function ## para que no tire error

  def train(self, X, y):
    X = add_bias(X, self.bias)
    n, p = X.shape
    weights = np.zeros(shape=(p, 1))

    i = 0
    error = 0
    error_min = 100000000
    w_min = weights
    
    while (error_min > 0.01 and i < self.epochs):
      # take a random sample
      mu = rng.integers(0, n)
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
    error = self.error_function(X, y, weights, self.g) ##podria ser error_min
    return self.weights, error, i

  def predict(self, X):
    if self.weights is not None:
      X = add_bias(X, self.bias)
      hs = h(X, self.weights)
      predictions = np.apply_along_axis(self.g, 1, hs)
      return predictions

# NO ESTAN FUNCIONANDO
# normalizo X, y

class LinearPerceptron(Perceptron):
  def __init__(self, learning_rate, epochs) -> None:
    super().__init__(learning_rate, epochs)
    self.g = id_function
    self.g_prime = one_function
    self.error_function = mean_squared_error
    self.bias = 1
    self.delta_w_f = delta_w_linear


class NonLinearPerceptron(Perceptron):
  def __init__(self, learning_rate, epochs) -> None:
    super().__init__(learning_rate, epochs)
    self.g = tanh
    self.g_prime = tanh_derivative
    self.error_function = mean_squared_error
    self.bias = 1
    self.delta_w_f = delta_w_nonlinear