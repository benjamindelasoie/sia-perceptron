
import utils
import numpy as np
from random import random
import matplotlib.pyplot as plt


class MultilayerPerceptron:

    def __init__(self, num_inputs=1, hidden_layers=[5], num_outputs=1):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [num_inputs] + hidden_layers + [num_outputs]

        weights = []
        for i in range(len(layers) - 1):
            w = np.zeros((layers[i], layers[i + 1]))# podría ser random sino
            weights.append(w)
        self.weights = weights

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations


    def forward_propagate(self, inputs):
        activations = inputs

        self.activations[0] = activations

        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)

            activations = self._sigmoid(net_inputs)

            self.activations[i + 1] = activations

        return activations


    def back_propagate(self, error):

        for i in reversed(range(len(self.derivatives))):

            activations = self.activations[i+1]

            delta = error * self._sigmoid_derivative(activations)

            delta_re = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]

            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            self.derivatives[i] = np.dot(current_activations, delta_re)

            error = np.dot(delta, self.weights[i].T)


    def train(self, inputs, targets, epochs, learning_rate):
        errors = []
        for i in range(epochs):
            sum_errors = 0

            for j, input in enumerate(inputs):
                target = targets[j]

                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                self.gradient_descent(learning_rate)

                sum_errors += self._mse(target, output)


            if i+1 % 100 == 0:
                print("Error: {} at epoch {}".format(sum_errors / len(inputs), i+1))

            errors.append(sum_errors)

        plt.plot(list(range(0,epochs)),errors)
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.show()

        print("Training complete!")
        print("=====")
        print(errors[-1])


    def gradient_descent(self, learningRate=1):

        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate


    def _sigmoid(self, x):

        y = 1.0 / (1 + np.exp(-x))
        return y


    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)


    def _mse(self, target, output):
        return np.average((target - output) ** 2)


if __name__ == "__main__":

    inputs = np.array(utils.data_import('TP2-ej3-digitos.txt', 7))
    outputs = np.array(utils.data_import('TP2-ej3-c-expected.txt', 1), dtype=float)


    print("inputs", inputs, type(inputs), inputs.size, inputs.shape)
    print("outputs", outputs, type(outputs), outputs.shape)

    mlp = MultilayerPerceptron(35, [10,15], 10)

    mlp.train(inputs, outputs, 50000, 0.01)

    for input in inputs:
        output = mlp.forward_propagate(input)


        print()
        print("Our network believes that {} = {} ".format(input, output))