from .base import Evaluator
from ..exception import ParameterException

import numpy as np


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
        return self

    def get_metric(self, metric):
        if metric == "mse":
            return self.mse()
        elif metric == "opposite_mse":
            return -self.mse()
        elif metric == "explained_variance":
            return self.explained_variance()
        else:
            raise ParameterException("metric {:s} unsupported.".format(metric))

    def mse(self):
        return np.mean(np.power(self.true - self.predict, 2))

    def explained_variance(self):
        return 1. - np.var(self.predict - self.true) / np.var(self.true)

    def r2(self):

        return 1 - np.sum(np.power(self.true - self.predict, 2)) / np.sum(np.power(self.true - self.true.mean(), 2))
