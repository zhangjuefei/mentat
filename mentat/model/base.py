import abc
import pandas as pd

from ..base import Operator
from ..dataframe import ZDataFrame


class Model(Operator):
    __metaclass__ = abc.ABCMeta

    def fit(self, data):
        self.train(data.features(), data.response())

    def evaluate(self, data):
        predict = self.predict(data.features())

        if data.response_encode:
            predict_category = pd.DataFrame(data.reverse_response(predict), columns=["predict_category"],
                                            index=data.data.index.copy())
            predict_prob = pd.DataFrame(predict, columns=data.category, index=data.data.index.copy())
            model_output = pd.merge(predict_category, predict_prob, left_index=True, right_index=True)
        else:
            model_output = pd.DataFrame(predict, columns=["predict_value"], index=data.data.index.copy())

        return ZDataFrame(
            pd.merge(model_output, data.data, left_index=True, right_index=True),
            response_column=data.response_column,
            ignores=data.ignores,
            response_encode=data.response_encode,
            category=data.category
        )

    @abc.abstractmethod
    def train(self, features, response):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass
