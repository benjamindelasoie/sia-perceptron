
import numpy as np
from layers import FinalLayer, Layer


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
        self.bias = 1



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





