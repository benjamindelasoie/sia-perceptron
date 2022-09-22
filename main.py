import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

from normalize import *
from perceptron import LinearPerceptron, NonLinearPerceptron, Perceptron, add_bias


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

X, y = datasets.make_blobs(n_samples=150,
                            n_features=2,
                            centers=2,
                            cluster_std=1.05,
                            random_state=2)

# fig = plt.figure(figsize=(10,8))
# plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r^')
# plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
# plt.xlabel("feature 1")
# plt.ylabel("feature 2")
# plt.title('Random Classification Data with 2 classes')

# y = np.where(y == 0, -1, 1).reshape(len(y), 1)


def run_simple():
  X, y = AND_X, AND_y

  p = Perceptron(20, 100)

  theta, error, n_epochs = p.train(X, y)
  print("theta\n", theta)
  print("finished training with error:", error)
  print("epochs", n_epochs)

  predictions = p.predict(X)
  print("predictions")
  print(predictions)

  error = np.abs(y - predictions)
  print(f"prediction error = {error}")

def run_linear():
  data = np.genfromtxt(fname="./TP2-ej2-conjunto.csv", skip_header=1, delimiter=',', dtype=np.float64)
  X, y = data[:, :-1], data[:,-1].reshape(len(data), 1)

  X = add_bias(X)
  # X_esc, y_esc = escale_all(X), escale_all(y)

  print("X")
  print(X)
  print("y")
  print(y)

  # sns.relplot(data, x=data[:,0], y=data[:,-1])
  # plt.show()
  # sns.relplot(data, x=data[:,1], y=data[:,-1])
  # plt.show()
  # sns.relplot(data, x=data[:,2], y=data[:,-1])
  # plt.show()

  n = 0.1
  for i in range(10):
    p = LinearPerceptron(n, 1000)
    weights, error =  p.train(X, y)
    
    print('_______________')
    print(n)
    print(weights)
    print(error)
    n = n/10

def run_nonlinear():
  pass

def main():
  run_simple()

if __name__ == "__main__":
  main()