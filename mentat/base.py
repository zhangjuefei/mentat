import abc


class Operator(object):
    __metaclass__ = abc.ABCMeta

    need_fit = True  # operator need fit
    need_evaluate = True  # operator need evaluate

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def evaluate(self, data):
        pass

    def set_need_fit(self, need_fit):
        self.need_fit = bool(need_fit)
        return self

    def set_need_evaluate(self, need_evaluate):
        self.need_evaluate = bool(need_evaluate)
        return self
