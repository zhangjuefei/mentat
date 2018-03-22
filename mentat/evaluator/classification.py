from .base import Evaluator
from ..exception import ParameterException

import pandas as pd


class ClassificationEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

        self.prediction = None
        self.cm = None

    def fit(self, data):
        self.prediction = data
        response_column = self.prediction.response_column
        self.cm = self.prediction.data.groupby([response_column, "predict_category"])[response_column].count().unstack(
            "predict_category").reindex(self.prediction.category, axis=1).fillna(0.).astype("int")

        self.metrics["accuracy"] = (self.prediction.data[response_column] == self.prediction.data[
            "predict_category"]).astype("int").sum() / self.prediction.shape()[0]

        cm_aug = self.cm.copy()
        cm_aug["total"] = cm_aug.sum(axis=1)
        cm_aug.columns.name = "predict"
        cm_aug = pd.concat([cm_aug, cm_aug.sum(axis=0).to_frame("total").transpose()], axis=0)
        cm_aug.index.name = "true"
        self.metrics["confusion_matrix"] = cm_aug

        precision = (self.cm.values.diagonal() / self.cm.sum(axis=0))
        recall = (self.cm.values.diagonal() / self.cm.sum(axis=1))
        f1_score = (precision * recall / 2 * (precision + recall))

        self.metrics["classification_report"] = pd.concat([
            precision.to_frame("precision"),
            recall.to_frame("recall"),
            f1_score.to_frame("f1 score")
        ], axis=1)

        return self

    def evaluate(self, data):
        return None
