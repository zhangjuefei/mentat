import abc


class Operator(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def evaluate(self, data):
        pass

