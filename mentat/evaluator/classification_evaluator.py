from .base import Evaluator
from ..exception import ParameterException

import pandas as pd


class ClassificationEvaluator(Evaluator):
    prediction = None
    cm = None

    def __init__(self):
        super().__init__()

    def fit(self, data):
        self.prediction = data
        response_column = self.prediction.response_column
        self.cm = self.prediction.data.groupby([response_column, "predict_category"])[response_column].count().unstack(
            "predict_category", fill_value=0).astype("int")

        return self

    def evaluate(self, data):
        return None

    def get_metric(self, metric):
        if metric == "accuracy":
            return self.accuracy()
        else:
            raise ParameterException("metric {:s} unsupported.".format(metric))

    def confusion_matrix(self):
        cm = self.cm.copy()
        cm["total"] = cm.sum(axis=1)
        cm.columns.name = "predict"
        cm = pd.concat([cm, cm.sum(axis=0).to_frame("total").transpose()], axis=0)
        cm.index.name = "true"
        return cm

    def accuracy(self):
        response_column = self.prediction.response_column
        return (self.prediction.data[response_column] == self.prediction.data["predict_category"]).astype("int").sum() / \
               self.prediction.shape()[0]

    def report(self):
        cm = self.cm
        precision = (cm.values.diagonal() / cm.sum(axis=0))
        recall = (cm.values.diagonal() / cm.sum(axis=1))
        f1_score = (precision * recall / 2 * (precision + recall))

        return pd.concat([
            precision.to_frame("precision"),
            recall.to_frame("recall"),
            f1_score.to_frame("f1 score")
        ], axis=1)
