import matplotlib.pyplot as plt
import numpy as np

def plot_simple_perceptron(X, y, theta, title="Classification of 2 classes"):
  ymin, ymax = -5, 5
  print(f"ymin = {ymin} ymax = {ymax}")
  u = theta[0]
  w = theta[1:]
  a = -w[0] / w[1]
  xx = np.linspace(ymin, ymax)
  yy = a * xx + (u[0] / w[1])

  fig = plt.figure(figsize=(10, 8))
  plt.plot(X[:, 0][y == -1], X[:, 1][y == -1], 'r^')
  plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
  plt.xlabel("feature 1")
  plt.ylabel("feature 2")
  plt.title(title)
  plt.plot(xx, yy, 'k-')
  plt.show()



def plot_error(errors_means, stds, epochs):

  print("means", errors_means)
  print("stds", stds)

  plt.xlabel("Epochs")
  plt.ylabel("Error")
  plt.fill_between(epochs, errors_means - stds, errors_means + stds, color="lightblue", alpha=0.2)
  plt.plot(epochs, errors_means)


  plt.show()
