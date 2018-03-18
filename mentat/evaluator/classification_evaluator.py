from .base import Evaluator

import pandas as pd


class ClassificationEvaluator(Evaluator):
    prediction = None

    def __init__(self):
        super().__init__()

    def fit(self, data):
        self.prediction = data

    def evaluate(self, data):
        return None

    def confusion_matrix(self):
        response_column = self.prediction.response_column
        cm = self.prediction.data.groupby([response_column, "predict_category"])[response_column].count().unstack(
            "predict_category", fill_value=0).astype("int")

        cm["total"] = cm.sum(axis=1)
        cm.columns.name = "true/predict"
        return pd.concat([cm, cm.sum(axis=0).to_frame("total").transpose()], axis=0)

    def accuracy(self):
        response_column = self.prediction.response_column
        return (self.prediction.data[response_column] == self.prediction.data["predict_category"]).astype("int").sum() / \
               self.prediction.shape()[0]
