import copy
import numpy as np
import pandas as pd

from .base import Preprocessor


class Dummy(Preprocessor):
    def __init__(self):
        super().__init__()

        self.dummy_cols = None

    def fit(self, dataframe, columns=None):

        if columns is None:
            self.dummy_cols = dataframe.data[dataframe.feature_cols].select_dtypes(include=["object"]).columns
        else:
            self.dummy_cols = pd.Index(columns)

    def evaluate(self, dataframe):
        dataframe = copy.deepcopy(dataframe)

        for col in self.dummy_cols.values:
            distinct_values = dataframe.data[col].unique()
            values = dataframe.data[col].apply(lambda v: list(distinct_values).index(v))
            dummies = np.zeros((dataframe.shape()[0], len(distinct_values)))
            dummies[np.arange(dummies.shape[0]), values] = 1
            new_colomns = pd.Index(["{}_{:d}".format(col, i + 1) for i in range(len(distinct_values))])
            dummies = pd.DataFrame(dummies, columns=new_colomns)
            dataframe.data = pd.concat([dataframe.data, dummies], axis=1).drop(col, axis=1)
            dataframe.feature_cols = dataframe.feature_cols.union(new_colomns).difference([col])

        return dataframe
