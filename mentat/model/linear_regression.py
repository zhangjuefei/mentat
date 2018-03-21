import numpy as np
from .base import Model
from ..exception import UnSupportException


class LinearRegression(Model):
    methods = ["analytic", "sgd"]
    penalties = ["L2"]

    def __init__(self, method="analytic", eta=0.1, penalty="L2", regularization=0, max_epochs=10, threshold=1e-5, minibatch_size=5,
                 momentum=0.9, decay_power=0.5, verbose=False):
        super().__init__()

        method = str(method)
        if method not in self.methods:
            raise UnSupportException("method: {:s}".format(method))
        else:
            self.method = method

        penalty = str(penalty)
        if penalty not in self.penalties:
            raise UnSupportException("penalty: {:s}".format(penalty))
        else:
            self.penalty = penalty

        self.eta = float(eta)
        self.regularization = float(regularization)
        self.max_epochs = int(max_epochs)
        self.threshold = float(threshold)
        self.minibatch_size = int(minibatch_size)
        self.momentum = float(momentum)
        self.decay_power = float(decay_power)
        self.verbose = verbose

    def train(self, features, response):
        # x = np.mat(np.c_[[1.0] * features.shape[0], features])
        x = np.mat(np.c_[[1.0] * features.shape[0], features])
        y = np.mat(response)
        I = np.eye(x.shape[1])
        self.beta = np.mat(np.random.random((x.shape[1], 1)) / 100)

        if self.method == "analytic":
            self.beta = (x.T * x + self.regularization * I).I * x.T * y
        elif self.method == "sgd":

            accu_gradient = np.mat(np.zeros(self.beta.shape))

            epochs = 0
            iterations = 0
            start = 0
            train_set_size = x.shape[0]
            loss = []

            effective_eta = self.eta
            while True:

                end = start + self.minibatch_size
                minibatch_x = x[start:end]
                minibatch_y = y[start:end]
                start = (start + self.minibatch_size) % train_set_size

                y_predict = minibatch_x * self.beta
                gradient = -minibatch_x.T * (minibatch_y - y_predict) / len(minibatch_y) + self.regularization * self.beta / 100.
                accu_gradient = accu_gradient * self.momentum + effective_eta * gradient
                self.beta = self.beta - accu_gradient

                loss.append(float((minibatch_y - y_predict).T * (minibatch_y - y_predict)) / len(minibatch_y))

                iterations += 1
                # decay the learning rate
                effective_eta = self.eta / np.power(iterations, self.decay_power)

                if iterations % train_set_size == 0:
                    epochs += 1
                    mean_loss = np.mean(loss)
                    loss = []

                    if self.verbose:
                        print("epoch: {:d}. mean loss: {:.6f}. learning rate: {:.8f}".format(epochs, mean_loss,
                                                                                             effective_eta))

                    if epochs >= self.max_epochs or mean_loss < self.threshold:
                        break
        else:
            raise UnSupportException("optimizaation method {:s}".format(self.method))

    def predict(self, x):
        x = np.mat(np.c_[[1.0] * x.shape[0], x])
        return (x * self.beta).A

    def intercept(self):
        return self.beta.A1[0]

    def weights(self):
        return self.beta.A1[1:]

    def batch_loss(self, x, y):
        return
