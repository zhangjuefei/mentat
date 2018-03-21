import numpy as np
from .base import Model
from ..exception import UnSupportException


class LogisticRegression(Model):
    def __init__(self, method="Gradient", eta=0.1, threshold=1e-5, max_epochs=10, regularization=0.1, minibatch_size=5,
                 momentum=0.9, decay_power=0.5,
                 verbose=False):
        Model.__init__(self)

        self.method = str(method)  # IRLS, Newton or Gradient
        self.eta = float(eta)  # learning rate for Newton and Gradient method
        self.threshold = float(threshold)  # stopping loss threshold
        self.max_epochs = int(max_epochs)  # max number of iterations
        self.regularization = float(regularization)  # L2 regularization strength, only for Gradient Descent
        self.minibatch_size = int(minibatch_size)  # minibatch size
        self.decay_power = float(decay_power)  # learning rate decaying power for gradient descent and newton
        self.momentum = float(momentum)  # gradient descent momentum
        self.verbose = verbose

        self.beta = None

    def train(self, features, response):
        x = np.mat(np.c_[[1.0] * features.shape[0], features])
        y = np.mat(response)
        I = np.eye(x.shape[1])
        self.beta = np.mat(np.random.random((x.shape[1], 1)) / 100)
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

            linear_combination = -np.matmul(minibatch_x, self.beta)
            linear_combination[linear_combination > 1e2] = 1e2  # prevent overflow
            p = 1.0 / (1.0 + np.power(np.e, linear_combination))
            p[p >= 1.0] = 1.0 - 1e-10
            p[p <= 0.0] = 1e-10  # truncate the probabilities to prevent W from being singular
            w = np.mat(np.diag(np.multiply(p, 1.0 - p).A1))

            if self.method == "IRLS":
                z = minibatch_x * self.beta + w.I * (minibatch_y - p)
                self.beta = (minibatch_x.T * w * minibatch_x + 1e-10 * I).I * minibatch_x.T * w * z
            elif self.method == "Newton":
                self.beta = self.beta + effective_eta * (minibatch_x.T * w * minibatch_x + 1e-10 * I).I * minibatch_x.T * (minibatch_y - p)
            elif self.method == "Gradient":
                accu_gradient = accu_gradient * self.momentum + effective_eta * (
                        -minibatch_x.T * (minibatch_y - p) + self.regularization * self.beta)
                self.beta = self.beta - accu_gradient
            else:
                raise UnSupportException("optimizaation method {:s}".format(self.method))

            loss.append(float(-minibatch_y.T * np.log(p) - (1 - minibatch_y).T * np.log(1 - p)) / len(minibatch_y))

            iterations += 1
            # decay the learning rate
            effective_eta = self.eta / np.power(iterations, self.decay_power)

            if iterations % train_set_size == 0:
                epochs += 1
                mean_loss = np.mean(loss)
                loss = []

                if self.verbose:
                    print("epoch: {:d}. mean loss: {:.6f}. learning rate: {:.8f}".format(epochs, mean_loss, effective_eta))

                if epochs >= self.max_epochs or mean_loss < self.threshold:
                    break

    def predict(self, x):
        x = np.mat(np.c_[[1.0] * x.shape[0], x])
        linear_combination = -np.matmul(x, self.beta)
        linear_combination[linear_combination > 1e2] = 1e2  # prevent overflow
        p = 1.0 / (1.0 + np.power(np.e, linear_combination))

        return np.c_[1 - p.A1, p.A1]

    def intercept(self):
        return self.beta.A1[0]

    def weights(self):
        return self.beta.A1[1:]

    def batch_loss(self, x, y):
        return
