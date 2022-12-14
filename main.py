import json
import random
import numpy as np
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt

from multilayer_perceptron import MultilayerPerceptron
from normalize import *
from perceptron import Perceptron, LinearPerceptron, NonLinearPerceptron
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

XOR_X = np.array([
    [-1, 1],
    [1, -1],
    [-1, -1],
    [1, 1]])

XOR_y = np.array([
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


def run_simple(epochs, learning_rate, error_min):
    X = RANDOM_X[:120, :]
    y = RANDOM_y[:120]
    X_hat = RANDOM_X[120:, :]
    y_hat = RANDOM_y[120:]
    y = y.reshape(len(y), )
    y_hat = y_hat.reshape(len(y_hat), )

    print("X", X, X.shape)
    print("y", y, y.shape)
    print("X_hat", X_hat, X_hat.shape)
    print("y_hat", y_hat, y_hat.shape)

    p = Perceptron(learning_rate, epochs, error_min)

    theta, error, n_epochs = p.train(X, y.reshape(len(y), 1))
    print("theta\n", theta)
    print("finished training with error:", error)
    print("epochs:", n_epochs)

    utils.plot_simple_perceptron(X, y, theta, title="Training")

    pred = p.predict(X_hat)
    utils.plot_simple_perceptron(X_hat, pred.reshape(len(pred), ), p.weights, title="Testing")


def run_linear(epochs, learning_rate, error_min):
    data = np.genfromtxt(fname="./TP2-ej2-conjunto.csv", skip_header=1, delimiter=',', dtype=np.float64)
    print(data)

    # extract features and labels (as column data)
    X, y = data[:, :-1], data[:, -1][:, np.newaxis]

    p = LinearPerceptron(learning_rate, epochs, error_min)

    theta, error, epochs = p.train(X, y)
    print("theta", theta)
    print("error", error)
    print("epochs", epochs)
    os = p.predict(X)
    print("os", os)
    print("y", y)


def run_nonlinear(epochs, learning_rate, error_min):
    data = np.genfromtxt(fname="./TP2-ej2-conjunto.csv", skip_header=1, delimiter=',', dtype=np.float64)
    print(data)

    # extract features and labels (as column data)
    X, y = data[:, :-1], data[:, -1][:, np.newaxis]

    p = NonLinearPerceptron(learning_rate, epochs, error_min)

    y_norm = escale_all(y)

    theta, error, epochs = p.train(X, y_norm)
    print("theta", theta)
    print("error", error)
    print("epochs", epochs)
    os = p.predict(X)
    print("os", os)
    print("y", y_norm)

    error_abs = np.abs(y_norm - os)
    print("error_abs", error_abs)
    mean_error = np.mean(error_abs)
    print("generalize_error", mean_error)


def run_multi(input_file, expected_file):
    inputs = np.array(utils.data_import(input_file, 7))
    outputs = np.array(utils.data_import(expected_file, 1), dtype=float)


    print("inputs", inputs, type(inputs), inputs.size, inputs.shape)
    print("outputs", outputs, type(outputs), outputs.shape)

    mlp = MultilayerPerceptron(35, list(np.repeat(35, 15)), 10)

    mlp.train(inputs, outputs, 50000, 0.001)

    for input in inputs:
        # print(input)
        output = mlp.forward_propagate(input)


        print()
        print("Our network believes that {} = {} ".format(input, output))


def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    perceptron = config['perceptron']
    simple = perceptron['simple']
    linear = perceptron['lineal']
    non_linear = perceptron['no-lineal']
    multilayer = perceptron['multilayer']

    if simple['activated']:
        run_simple(simple['epochs'], simple['learning_rate'], simple['error_min'])
    if linear['activated']:
        run_linear(linear['epochs'], linear['learning_rate'], linear['error_min'])
    if non_linear['activated']:
        run_nonlinear(non_linear['epochs'], non_linear['learning_rate'], non_linear['error_min'])
    if multilayer['activated']:
        run_multi(multilayer['input_file'], multilayer['expected_file'])


if __name__ == "__main__":
    main()
