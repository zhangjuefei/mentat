import copy
import numpy as np
import pandas as pd

from .base import Preprocessor


class Dummy(Preprocessor):
    def __init__(self):
        super().__init__()

        self.dummy_cols = None

    def fit(self, data, columns=None):

        if columns is None:
            self.dummy_cols = data.data[data.feature_cols].select_dtypes(include=["object"]).columns
        else:
            self.dummy_cols = pd.Index(columns)

    def evaluate(self, data):
        data = copy.deepcopy(data)

        for col in self.dummy_cols.values:
            distinct_values = data()[col].unique()
            values = np.array(
                [int(np.where(distinct_values == v)[0][0]) for v in data()[col]])
            dummies = np.zeros((data.shape()[0], len(distinct_values)))
            dummies[np.arange(dummies.shape[0]), values] = 1
            new_colomns = pd.Index(["{}_{:d}".format(col, i + 1) for i in range(len(distinct_values))])
            dummies = pd.DataFrame(dummies, columns=new_colomns)
            data.data = pd.concat([data(), dummies], axis=1).drop(col, axis=1)
            data.feature_cols = data.feature_cols.union(new_colomns).difference([col])

        return data
