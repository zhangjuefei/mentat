from .base import Preprocessor
from ..exception import ParameterException


class RobustScale(Preprocessor):
    def __init__(self):
        super().__init__()

        self.median = None
        self.scale = None

    def fit(self, data, columns=None):
        if columns is None:
            self.median = data.data[data.feature_cols].select_dtypes(exclude="object").mean()
            self.scale = data.data[data.feature_cols].select_dtypes(exclude="object").std()
        else:

            if (data.data.dtypes[columns] == "object").any():
                raise ParameterException("contains non-numeric column")

            self.median = data.data[columns].mean()
            self.scale = data.data[columns].std()

    def evaluate(self, x):
        return (x - self.median) / self.scale
