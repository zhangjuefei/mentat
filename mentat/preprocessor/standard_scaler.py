from .base import Preprocessor
from ..exception import ParameterException
import copy


class StandardScaler(Preprocessor):
    def __init__(self):
        super().__init__()

        self.mean = None
        self.std = None

    def fit(self, data, columns=None):

        if columns is None:
            self.mean = data.data[data.feature_cols].select_dtypes(exclude="object").mean()
            self.std = data.data[data.feature_cols].select_dtypes(exclude="object").std()
        else:

            if (data.data.dtypes[columns] == "object").any():
                raise ParameterException("contains non-numeric column")

            self.mean = data.data[columns].mean()
            self.std = data.data[columns].std()

    def evaluate(self, data):
        data = copy.deepcopy(data)
        data.data[self.mean.index] = (data.data[self.mean.index] - self.mean) / self.std
        return data
