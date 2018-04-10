import numpy as np
from ..exception import UnSupportException, ParameterException
from .base import Model
from ..util import ParamValidator


class DNN(Model):

    pv = ParamValidator(
        {
            "input_shape": {"type": int},
            "shape": {"type": list},
            "activations" : {"type": list},
            "eta": {"type": [int, float]},
            "threshold": {"type": [int, float]},
            "softmax": {"type": bool},
            "max_epochs": {"type": int},
            "regularization": {"type": [int, float]},
            "minibatch_size": {"type": int},
            "momentum": {"type": [int, float], "range": [0.0, 1.0]},
            "decay_power": {"type": [int, float]},
            "verbose": {"type": bool},
        }
    )

    def __init__(self, input_shape, shape, activations, eta=0.5, threshold=1e-5, softmax=False, max_epochs=20,
                 regularization=0, minibatch_size=20, momentum=0.9, decay_power=0.2, verbose=False):
        Model.__init__(self)

        if not len(shape) == len(activations):
            raise ParameterException("activations must equal to number od layers.")

        self.pv("input_shape", input_shape)
        self.depth = len(self.pv("shape", shape))
        self.activity_levels = [np.mat([0])] * self.depth
        self.outputs = [np.mat(np.mat([0]))] * (self.depth + 1)
        self.deltas = [np.mat(np.mat([0]))] * self.depth

        self.eta = self.pv("eta", eta)
        self.effective_eta = self.eta
        self.threshold = self.pv("threshold", threshold)
        self.max_epochs = self.pv("max_epochs", max_epochs)
        self.regularization = self.pv("regularization", regularization)
        self.is_softmax = self.pv("softmax", softmax)
        self.verbose = self.pv("verbose", verbose)
        self.minibatch_size = self.pv("minibatch_size", minibatch_size)
        self.momentum = self.pv("momentum", momentum)
        self.decay_power = self.pv("decay_power", decay_power)
        self.iterations = 0
        self.epochs = 0

        self.activations = self.pv("activations", activations)
        self.activation_func = []
        self.activation_func_diff = []
        for f in activations:

            if f == "sigmoid":
                self.activation_func.append(np.vectorize(self.sigmoid))
                self.activation_func_diff.append(np.vectorize(self.sigmoid_diff))
            elif f == "identity":
                self.activation_func.append(np.vectorize(self.identity))
                self.activation_func_diff.append(np.vectorize(self.identity_diff))
            elif f == "relu":
                self.activation_func.append(np.vectorize(self.relu))
                self.activation_func_diff.append(np.vectorize(self.relu_diff))
            else:
                raise UnSupportException("activation function {:s}".format(f))

        self.weights = [np.mat(np.mat([0]))] * self.depth
        self.biases = [np.mat(np.mat([0]))] * self.depth
        self.acc_weights_delta = [np.mat(np.mat([0]))] * self.depth
        self.acc_biases_delta = [np.mat(np.mat([0]))] * self.depth

        self.weights[0] = np.mat(np.random.random((shape[0], input_shape)) / 100)
        self.biases[0] = np.mat(np.random.random((shape[0], 1)) / 100)
        for idx in np.arange(1, len(shape)):
            self.weights[idx] = np.mat(np.random.random((shape[idx], shape[idx - 1])) / 100)
            self.biases[idx] = np.mat(np.random.random((shape[idx], 1)) / 100)

    def compute(self, x):
        result = x
        for idx in np.arange(0, self.depth):
            self.outputs[idx] = result
            al = self.weights[idx] * result + self.biases[idx]
            self.activity_levels[idx] = al
            result = self.activation_func[idx](al)

        self.outputs[self.depth] = result
        return self.softmax(result) if self.is_softmax else result

    def predict(self, features):

        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ParameterException("data is empty.")

        return self.compute(np.mat(features).T).T.A

    def bp(self, d):
        tmp = d.T

        for idx in np.arange(0, self.depth)[::-1]:
            delta = np.multiply(tmp, self.activation_func_diff[idx](self.activity_levels[idx]).T)
            self.deltas[idx] = delta
            tmp = delta * self.weights[idx]

    def update(self):

        for idx in np.arange(0, self.depth):
            # current gradient
            weights_grad = -self.deltas[idx].T * self.outputs[idx].T / self.deltas[idx].shape[0] + \
                           self.regularization * self.weights[idx]
            biases_grad = -np.mean(self.deltas[idx].T, axis=1) + self.regularization * self.biases[idx]

            # accumulated delta
            self.acc_weights_delta[idx] = self.acc_weights_delta[
                                              idx] * self.momentum - self.effective_eta * weights_grad
            self.acc_biases_delta[idx] = self.acc_biases_delta[idx] * self.momentum - self.effective_eta * biases_grad

            self.weights[idx] = self.weights[idx] + self.acc_weights_delta[idx]
            self.biases[idx] = self.biases[idx] + self.acc_biases_delta[idx]

    def train(self, features, response):

        if features.shape[0] == 0 or features.shape[1] == 0 or features.shape[0] != response.shape[0] or response.shape[
            1] == 0:
            raise ParameterException("features or response is empty or number of instances is not equal")

        x = np.mat(features)
        y = np.mat(response)
        loss = []
        self.iterations = 0
        self.epochs = 0
        start = 0
        train_set_size = x.shape[0]

        while True:

            end = start + self.minibatch_size
            minibatch_x = x[start:end].T
            minibatch_y = y[start:end].T
            start = (start + self.minibatch_size) % train_set_size

            yp = self.compute(minibatch_x)
            d = minibatch_y - yp

            self.bp(d)
            self.update()

            if self.is_softmax:
                loss.append(np.mean(-np.sum(np.multiply(minibatch_y, np.log(yp + 1e-1000)), axis=0)))
            else:
                loss.append(np.mean(np.sqrt(np.sum(np.power(d, 2), axis=0))))

            self.iterations += 1

            # decay the learning rate
            self.effective_eta = self.eta / np.power(self.iterations, self.decay_power)

            if self.iterations % train_set_size == 0:
                self.epochs += 1
                mean_e = np.mean(loss)
                loss = []

                if self.verbose:
                    print("epoch: {:d}. mean loss: {:.6f}. learning rate: {:.8f}".format(self.epochs, mean_e,
                                                                                         self.effective_eta))

                if self.epochs >= self.max_epochs or mean_e < self.threshold:
                    break

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.power(np.e, min(-x, 1e2)))

    @staticmethod
    def sigmoid_diff(x):
        return np.power(np.e, min(-x, 1e2)) / (1.0 + np.power(np.e, min(-x, 1e2))) ** 2

    @staticmethod
    def relu(x):
        return x if x > 0 else 0.0

    @staticmethod
    def relu_diff(x):
        return 1.0 if x > 0 else 0.0

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def identity_diff(x):
        return 1.0

    @staticmethod
    def softmax(x):
        x[x > 1e2] = 1e2
        ep = np.power(np.e, x)
        return ep / np.sum(ep, axis=0)
