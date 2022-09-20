import numpy as np
from perceptron import LinearPerceptron, NonLinearPerceptron, Perceptron
import matplotlib.pyplot as plt
from sklearn import datasets

and_X = np.array([
  [-1, 1],
  [1, -1],
  [-1, -1],
  [1, 1]])

and_y = np.array([
  [-1],
  [-1],
  [-1],
  [1]
])

or_X = np.array([
  [-1, 1],
  [1, -1],
  [-1, -1],
  [1, 1]])

or_y = np.array([
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

fig = plt.figure(figsize=(10,8))
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title('Random Classification Data with 2 classes')

y = np.where(y == 0, -1, 1).reshape(len(y), 1)


def main():

  data = np.genfromtxt(fname="./TP2-ej2-conjunto.csv", skip_header=1, delimiter=',', dtype=np.float64)
  X, y = data[:, :-1], data[:,-1].reshape(len(data), 1)

  p = LinearPerceptron(1, 3)

  error = p.train(X, y)
  print("FINALIZO, ERROR:")
  print(error)

if __name__ == "__main__":
  main()