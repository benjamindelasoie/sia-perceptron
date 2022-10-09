from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class MultilayerPerceptron:

    def __init__(self, train_data, target, learning_rate=0.1, epochs=100, input=2, hidden=2, output=1):
        self.train_data = train_data
        self.target = target
        self.lr = learning_rate
        self.epochs = epochs

        # weight between input and hidden layer
        self.weights_ih = np.random.uniform(size=(input, hidden))

        # weight between hidden and output layer
        self.weights_ho = np.random.uniform(size=(hidden, output))

        # bias for the hidden layer
        self.bh = np.random.uniform(size=(1, hidden))

        # bias for the output layer
        self.bo = np.random.uniform(size=(1, output))

        self.losses = []
        self.errors0 = []
        self.errors1 = []
        self.errors2 = []
        self.errors3 = []

    def backward(self):
        loss = 0.5 * (self.target - self.output_final) ** 2
        self.losses.append(np.sum(loss))

        error_term = (self.target - self.output_final)
        self.errors0.append(error_term[0])
        self.errors1.append(error_term[1])
        self.errors2.append(error_term[2])
        self.errors3.append(error_term[3])

        grad_hidden = self.train_data.T @ (
                    ((error_term * self.sigmoid_derivative(self.output_final)) * self.weights_ho.T) * self.sigmoid_derivative(
                self.hidden_out))

        grad_outer = self.hidden_out.T @ (error_term * self.sigmoid_derivative(self.output_final))

        self.weights_ih += self.lr * grad_hidden
        self.weights_ho += self.lr * grad_outer

        self.bh += np.sum(
            self.lr * ((error_term * self.sigmoid_derivative(self.output_final)) * self.weights_ho.T) * self.sigmoid_derivative(
                self.hidden_out), axis=0)
        self.bo += np.sum(self.lr * error_term * self.sigmoid_derivative(self.output_final), axis=0)

    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh_function(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - (np.tanh(x)) ** 2

    def forward(self, training_data):
        self.hidden = np.dot(training_data, self.weights_ih) + self.bh
        self.hidden_out = self.sigmoid_function(self.hidden)
        self.output = np.dot(self.hidden_out, self.weights_ho) + self.bo
        self.output_final = self.sigmoid_function(self.output)
        return self.output_final

    def classify(self, datapoint):
        datapoint = np.transpose(datapoint)
        if self.forward(datapoint) >= 0.5:
            return 1
        return 0

    def plot(self, h=0.01):
        sns.set_style('darkgrid')
        plt.figure()

        plt.axis('scaled')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)

        colors = {
            0: "ro",
            1: "go"
        }

        for i in range(len(self.train_data)):
            plt.plot([self.train_data[i][0]],
                     [self.train_data[i][1]],
                     colors[self.target[i][0]],
                     markersize=20)

        x_range = np.arange(-0.1, 1.1, h)
        y_range = np.arange(-0.1, 1.1, h)

        xx, yy = np.meshgrid(x_range, y_range, indexing='ij')
        Z = np.array([[self.classify([x, y]) for x in x_range] for y in y_range])

        plt.contourf(xx, yy, Z, colors=['red', 'green', 'green', 'blue'], alpha=0.4)

    def plot_errors(self):
        epochs = list(range(0, len(self.errors0)))
        plt.plot(epochs, self.errors0, label="Input 1")
        plt.plot(epochs, self.errors1, label="Input 2")
        plt.plot(epochs, self.errors2, label="Input 3")
        plt.plot(epochs, self.errors3, label="Input 4")
        plt.legend()
        plt.show()

    def train(self):
        for _ in range(self.epochs):
            self.forward(self.train_data)
            self.backward()


training_xor = np.array(
    [
        [0, 1],
        [1, 0],
        [0, 0],
        [1, 1]])

target_xor = np.array(
    [
        [1],
        [1],
        [0],
        [0]])


mlp = MultilayerPerceptron(training_xor, target_xor, 0.2, 5000)
mlp.train()
mlp.plot_errors()
mlp.plot()

