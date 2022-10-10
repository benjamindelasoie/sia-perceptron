import numpy as np

def tanh(x):
  beta = 0.1
  return np.tanh(beta * x)


def tanh_derivative(x):
  beta = 0.1
  return beta * (1 - (tanh(beta * x))**2)

class Layer:
    def __init__(self, previous_layer_size, post_layer_size, lr) -> None:
        self.W = np.random.random((previous_layer_size, post_layer_size))
        self.g = tanh
        self.bias = 1
        self.g_prime = tanh_derivative  ## para que no tire error
        self.lr = lr

    def get_output(self, X):
        self.V_prev = X
        aux = self.W.T
        h = np.zeros((len(X), len(aux)))

        for i in range(len(aux)):
            for j in range(len(X)):
                h[j][i] = np.dot(aux[i], X[j])

        #### cada fila corresponde a un mu cada columna corresponde a cada neurona de la capa posterior
        # print(output)
        self.h = h
        return self.g(h)

    def update_weights(self, delta_prev):
        #print("h", self.h)

        g_primes = self.g_prime(self.h)


        #print("W", self.W)
        #print("delta", delta_prev)

        aux = self.W.T
        sum_ = np.zeros((len(delta_prev), len(aux)))

        #print(aux[0], delta_prev[0])

        for i in range(len(aux)):
            for j in range(len(delta_prev)):
                mult = aux[i] * delta_prev[j]
                #print("mult", mult)
                sum_[j][i] = np.sum(mult)
                #print(sum_[j][i])

        #print("sum", sum_)
        #print("g_primes", g_primes)

        delta = g_primes * sum_

        delta_w = np.zeros(self.W.shape)

        deltaT = delta.T
        V = self.V_prev.T

        for i in range(len(delta_w)):
            for j in range(len(delta_w[i])):
                delta_w[i][j] = self.lr * np.sum(deltaT[j]*V[i])


        self.W += delta_w
        #print("W", self.W)

        return delta









class FinalLayer(Layer):
    def __init__(self, previous_layer_size, post_layer_size, lr) -> None:
        super().__init__(previous_layer_size, post_layer_size, lr)

    def get_output(self, X):
        self.V_prev = X
        aux = self.W.T
        h = np.zeros((len(X), len(aux)))

        for i in range(len(aux)):
            for j in range(len(X)):
                h[j][i] = np.dot(aux[i], X[j])

        #### cada fila corresponde a un mu cada columna corresponde a cada neurona de la capa posterior
        #print(output)
        self.h = h
        self.o = self.g(h)
        return self.o

    def update_weights(self, y):

        #print("h", self.h)

        g_primes = self.g_prime(self.h)

        #print("g_primes", g_primes)

        delta = (y - self.o) * g_primes
        #print("delta", delta)

        #print("V_prev", self.V_prev)
        delta_w = delta * self.V_prev

        #print("deltaW", delta_w)

        delta_w = np.sum(delta_w, axis=0) ## habria que ver sacarle la media

        #print("deltaW", delta_w)
        #print("W", self.W)
        self.W += self.lr * delta_w.reshape(self.W.shape)
        #print("W", self.W)

        return delta



