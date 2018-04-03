from threading import Thread

import abc

from ..base import Operator


class Trainer(Operator):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super().__init__()
        self.best_model = None

    def fit(self, data):
        self.train(data)

    def evaluate(self, data):
        if self.best_model is None:
            return None

        return self.best_model.evaluate(data)

    @abc.abstractmethod
    def train(self, data):
        pass


class MultiTrainer(Trainer):
    __metaclass__ = abc.ABCMeta

    def __init__(self, train_fraction=0.75, evaluator=None, metric=None):
        super().__init__()

        self.train_fraction = train_fraction
        self.evaluator = evaluator
        self.metric = metric
        self.metrics = {}
        self.evaluators = {}

    @abc.abstractmethod
    def train(self, data):
        pass

    class SingleTrainer(Thread):

        def __init__(self, model, data):
            Thread.__init__(self)
            self.model = model
            self.data = data

        def run(self):
            self.model.fit(self.data)

    def train_models(self, models, data):
        threads = []
        for model in models:
            thread = self.SingleTrainer(model, data)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
