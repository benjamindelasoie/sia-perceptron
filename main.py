import numpy as np
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from normalize import *
from perceptron import LinearPerceptron, NonLinearPerceptron, Perceptron
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

  X = RANDOM_X[:120,:]
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

  theta, error, n_epochs = p.train(X, y.reshape(len(y), 1))
  print("theta\n", theta)
  print("finished training with error:", error)
  print("epochs:", n_epochs)

  utils.plot_simple_perceptron(X, y, theta, title="Training")

  pred = p.predict(X_hat)
  utils.plot_simple_perceptron(X_hat, pred.reshape(len(pred),), p.weights, title="Testing")

def run_linear():
  data = np.genfromtxt(fname="./TP2-ej2-conjunto.csv", skip_header=1, delimiter=',', dtype=np.float64)
  print(data)

  # extract features and labels (as column data)
  X, y = data[:, :-1], data[:,-1][:, np.newaxis]

  # plot each feature against the label
  # for i in range(X.shape[1]):
  #   sns.relplot(data, x=data[:,i], y=data[:,-1])
  #   plt.show()

  p = LinearPerceptron(0.00001, 1000000)

  theta, error, epochs = p.train(X, y)
  print("theta", theta)
  print("error", error)
  print("epochs", epochs)
  os = p.predict(X)
  print("os", os)
  print("y-os", y-os)



def run_nonlinear():
  pass

def main():
  run_linear()

if __name__ == "__main__":
  main()