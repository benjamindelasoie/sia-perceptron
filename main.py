import random

import numpy as np
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from normalize import *
from perceptron import Perceptron, LinearPerceptron, NonLinearPerceptron
from perceptronMulticapa import PerceptronMulticapa
import utils

AND_X = np.array([
  [-1, 1],
  [1, -1],
  [-1, -1],
  [1, 1]])

AND_y = np.array([
  [-1],
  [-1],
  [-1],
  [1]
])

OR_X = np.array([
  [-1, 1],
  [1, -1],
  [-1, -1],
  [1, 1]])

OR_y = np.array([
  [1],
  [1],
  [-1],
  [-1]
])

RANDOM_X, RANDOM_y = datasets.make_blobs(n_samples=150,
                            n_features=2,
                            centers=2,
                            cluster_std=1.05,
                            random_state=2)
RANDOM_y = np.where(RANDOM_y == 0, -1, 1)

def run_simple():

  X = RANDOM_X[:120, :]
  y = RANDOM_y[:120]
  X_hat = RANDOM_X[120:, :]
  y_hat = RANDOM_y[120:]
  y = y.reshape(len(y),)
  y_hat = y_hat.reshape(len(y_hat),)

  print("X", X, X.shape)
  print("y", y, y.shape)
  print("X_hat", X_hat, X_hat.shape)
  print("y_hat", y_hat, y_hat.shape)

  p = Perceptron(1, 10000)

  theta, errors, epochs = p.train(X, y.reshape(len(y), 1))
  print("theta\n", theta)
  print("finished training with error:", errors[-1])
  print("epochs:", epochs[-1])

  utils.plot_simple_perceptron(X, y, theta, title="Training")

  pred = p.predict(X_hat)
  utils.plot_simple_perceptron(X_hat, pred.reshape(len(pred),), p.weights, title="Testing")

def run_linear():
  data = np.genfromtxt(fname="./TP2-ej2-conjunto.csv", skip_header=1, delimiter=',', dtype=np.float64)
  print(data)

  # extract features and labels (as column data)
  X, y = data[:, :-1], data[:, -1][:, np.newaxis]

  # plot each feature against the label
  # for i in range(X.shape[1]):
  #   sns.relplot(data, x=data[:,i], y=data[:,-1])
  #   plt.show()

  #X = np.arange(15)
  #print(X)
  #n = X.size
  #X = X.reshape(n, 1)
  #print(X)
  #y = X*2
  #print(y)

  max_epochs = 10
  iters = 100
  eta = 0.005

  error_mat = np.zeros((iters, max_epochs))
  for i in range(iters):
    p = LinearPerceptron(eta, max_epochs) #eta, un rango de ejemplo es entre 10-4 o 10-1.
    theta, errors, epochs = p.train(X, y)
    error_mat[i] = np.array(errors)

  print(error_mat)
  error_acum = np.sum(error_mat, axis=0)
  means = error_acum / iters

  stds = []

  for i in range(means.size):
    stds.append(np.sqrt(np.mean(error_mat.T[i] - means[i])**2))

  print("stds", stds)



  print("theta", theta)
  print("error", errors[-1])
  print("epochs", epochs[-1])
  os = p.predict(X)
  print("os", os)
  print("y", y)

  utils.plot_error(means, stds, epochs)



def run_nonlinear():
  data = np.genfromtxt(fname="./TP2-ej2-conjunto.csv", skip_header=1, delimiter=',', dtype=np.float64)
  print(data)

  # extract features and labels (as column data)
  X, y = data[:, :-1], data[:, -1][:, np.newaxis]

  # plot each feature against the label
  # for i in range(X.shape[1]):
  #   sns.relplot(data, x=data[:,i], y=data[:,-1])
  #   plt.show()

  max_epochs = 10
  iters = 100
  beta = 0.01
  eta = 0.005

  p = NonLinearPerceptron(eta, max_epochs, beta)

  y_norm = escale_all(y)
  #print("min", min(y))
  #print("max", max(y))

  #X = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
  #print(X)
  #n = X.size
  #X = X.reshape(n, 1)
  #print(X)
  #y = X*0.1
  #print(y)



  #y_norm= 2 * (y - min(y))/(max(y)-min(y)) - 1
  # y_norm = (y - min(y))/(max(y)-min(y))  para la funcion sigmoida

  theta, errors, epochs = p.train(X, y_norm)
  print("theta", theta)
  print("error", errors[-1])
  print("epochs", epochs[-1])
  os = p.predict(X)
  print("os", os)
  print("y", y_norm)

  error_abs = np.abs(y_norm - os)
  print("error_abs", error_abs)
  mean_error=np.mean(error_abs)
  print("generalize_error", mean_error)

  utils.plot_error(errors, epochs)



def run_multicapa_xor():
  xor= PerceptronMulticapa(0.005, 100000, 2, 2, 1)

  error, iter = xor.train(OR_X,OR_y)

  print("error", error)
  print("epochs", iter)

  print(xor.predict(OR_X))


def main():
  run_linear()



if __name__ == "__main__":
  main()