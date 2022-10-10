import random

import numpy as np
from layers import FinalLayer, Layer

rng = np.random.default_rng()


def h(X, weights):
  return np.dot(X, weights)

def id_function(x):
  return x

def one_function(x):
  return 1.

def tanh(x):
  beta = 100
  return np.tanh(beta * x)


def tanh_derivative(x):
  beta = 100
  return beta * (1 - (tanh(beta * x))**2)


def sigmoid_function(x):
  return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
  return x * (1 - x)


def mean_squared_error(y, os):
  return np.mean((y - os) ** 2)

def add_bias(X, value):
  n, *p = X.shape
  bias = np.repeat(value, n).reshape((n, 1))
  n = np.concatenate((bias, X), axis=1)
  return n

class PerceptronMulticapa:
    def __init__(self, learning_rate, epochs, layers_sizes) -> None:
        self.lr = learning_rate
        self.epochs = epochs
        self.layers_sizes = layers_sizes
        self.layers = []
        for i in range (layers_sizes.size - 1):
            if(i == layers_sizes.size - 2):
                self.layers.append(FinalLayer(layers_sizes[i], layers_sizes[i + 1], learning_rate))
            elif i == 0 :
                self.layers.append(Layer(layers_sizes[i] + 1, layers_sizes[i + 1], learning_rate))
            else:
                self.layers.append(Layer(layers_sizes[i], layers_sizes[i + 1], learning_rate))
        #    if i == 0:
        #        self.Ws.append(np.zeros((layers[i] + 1, layers[i + 1])))
        #    else:
        #        self.Ws.append(np.zeros(()))

        #self.W1 = np.zeros((input_size + 1, hidden_size)) # +1 para el bias
        #self.W2 = np.zeros((hidden_size, output_size))
        self.g = tanh
        self.bias = 1
        self.g_prime = tanh_derivative  ## para que no tire error



    def forward(self, X):
        previous = np.copy(X)
        for i in range(self.layers.__len__()):
            previous = self.layers[i].get_output(previous)

        return previous

    def backpropagation(self, X, y):

        os = self.forward(X)

        aux = (y - os) ** 2

        sum_error = np.sum(aux, axis=1)  ## suma de indice i por cada mu

        error = np.sum(sum_error) / 2  # sino np.mean()

        previous = y

        for i in reversed(range(self.layers.__len__())):
            #print("layer", i)
            previous = self.layers[i].update_weights(previous)

        return error


    def train(self, X, y):
        X = add_bias(X, self.bias)

        i = 0
        error = 0
        error_min = 100000000

        while (error_min > 0.005 and i < self.epochs):
            error = self.backpropagation(X, y)
            #print(error)
            if (error < error_min):
                error_min = error
            i = i + 1
        # error = self.error_function(X, y, weights, self.g)  ##podria ser error_min
        return error, i

    def predict(self, X):
        X = add_bias(X, self.bias)
        return self.forward(X)













    def ada(self,X):
        X = add_bias(X, self.bias)
        print("x", X)
        print("W1", self.W1)
        self.z1 = h(X, self.W1)
        self.a1 = self.g(self.z1)
        self.z2 = h(self.a1, self.W2)
        self.output = self.g(self.z2)
        return self.output

    def backprop(self, X, y):
        output = self.forward(X)
        error_out = output - y
        delta_out = error_out * self.g_prime(output)
        derivative_W2 = np.dot(self.a1.T, delta_out)
        error_hidden = np.dot(delta_out, self.W2.T)
        delta_hidden = error_hidden * self.g_prime(self.a1)
        derivative_W1 = np.dot(X.T, delta_hidden)
        # gradient descent
        self.W2 += derivative_W2 * self.lr
        self.W1 += derivative_W1 * self.lr
        return mean_squared_error(y, output)




