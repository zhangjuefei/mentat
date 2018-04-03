import numpy as np

from .base import Evaluator


class RegressionEvaluator(Evaluator):
    prediction = None
    true = None
    predict = None

    def __init__(self):
        super().__init__()

    def fit(self, data):
        self.prediction = data
        self.true = self.prediction.response().ravel()
        self.predict = self.prediction.data["predict_value"].values

        self.metrics["mse"] = np.mean(np.power(self.true - self.predict, 2))
        self.metrics["opposite_mse"] = -self.metrics["mse"]
        self.metrics["explained_variance"] = 1 - np.var(self.predict - self.true) / np.var(self.true)
        self.metrics["R2"] = 1 - np.sum(np.power(self.true - self.predict, 2)) / np.sum(
            np.power(self.true - self.true.mean(), 2))

        return self

    def evaluate(self, data):
        return None
