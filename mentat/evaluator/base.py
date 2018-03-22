import abc

from ..base import Operator


class Evaluator(Operator):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.metrics = {}

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def evaluate(self, data):
        pass

    def __getitem__(self, metric):
        return self.metrics[metric]
