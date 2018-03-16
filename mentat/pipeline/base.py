import abc
import copy

from ..base import Operator


class Pipeline(Operator):
    __metaclass__ = abc.ABCMeta

    operators = []

    def __init__(self):
        super().__init__()

    def fit(self, data):
        tmp = copy.deepcopy(data)
        for op in self.operators:
            op.fit(tmp)
            tmp = op.evaluate(tmp)

    def evaluate(self, data):
        tmp = copy.deepcopy(data)
        for op in self.operators:
            tmp = op.evaluate(tmp)

        return tmp

    def add_operator(self, operator):
        self.operators.append(operator)
