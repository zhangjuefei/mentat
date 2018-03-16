import abc
from ..base import Operator
from ..exception import ModelException


class Trainer(Operator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.best_model = None

    def fit(self, data):
        self.train(data)

    def evaluate(self, data):

        if self.best_model is None:
            raise ModelException("not trainned")

        return self.best_model.evaluate(data)

    @abc.abstractmethod
    def train(self, data):
        pass
