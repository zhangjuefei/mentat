import abc
import copy

from ..base import Operator
from ..exception import ParameterException


class Pipeline(Operator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, stages):
        super().__init__()

        if not isinstance(stages, dict) or len(stages) == 0:
            raise ParameterException("stages must be a dict.")

        self.stages = stages

    def fit(self, data):
        tmp = copy.deepcopy(data)
        for name, operator in self.stages.items():
            operator.fit(tmp)
            tmp = operator.evaluate(tmp)

    def evaluate(self, data):
        tmp = copy.deepcopy(data)
        for name, operator in self.stages.items():
            tmp = operator.evaluate(tmp)

        return tmp
