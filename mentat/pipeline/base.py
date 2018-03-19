import abc
import copy
from collections import OrderedDict

from ..base import Operator
from ..exception import ParameterException


class Pipeline(Operator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, stages):
        super().__init__()

        if not isinstance(stages, dict) or len(stages) == 0:
            raise ParameterException("stages must be a dict.")

        self.stages = OrderedDict(stages)

    def fit(self, data):
        tmp = copy.deepcopy(data)
        for name, operator in self.stages.items():

            if operator.need_fit:
                operator.fit(tmp)

            if operator.need_evaluate:
                tmp = operator.evaluate(tmp)

    def evaluate(self, data):
        tmp = copy.deepcopy(data)
        for name, operator in self.stages.items():

            if operator.need_evaluate:
                tmp = operator.evaluate(tmp)

        return tmp

    def get_operator(self, name):

        if self.stages[name]:
            return self.stages[name]
        else:
            raise ParameterException("name {:s} dosen't exist".format(name))
