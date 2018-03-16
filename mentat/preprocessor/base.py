import abc

from ..base import Operator


class Preprocessor(Operator):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def evaluate(self, data):
        pass

    def fit_evaluate(self, data):
        self.fit(data)
        return self.evaluate(data)
