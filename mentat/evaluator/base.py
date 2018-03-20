import abc

from ..base import Operator


class Evaluator(Operator):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def evaluate(self, data):
        pass

    @abc.abstractmethod
    def get_metric(self, metric):
        pass
