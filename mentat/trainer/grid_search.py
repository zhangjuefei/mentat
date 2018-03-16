from threading import Thread

from .base import Trainer


class GridSearchTrainer(Trainer):
    def __init__(self, model, train_fraction=0.75):
        Trainer.__init__(self, model)
        self.train_fraction = train_fraction

    class SingleTrainer(Thread):

        def __init__(self, model, train):
            Thread.__init__(self)
            self.model = model
            self.train = train

        def run(self):
            self.model.fit(self.train.features(), self.train.response())

    def train(self, data):
        train, test = data.split(self.train_fraction)

        threads = []
        models = [self.model]
        for model in self.models:
            thread = self.SingleTrainer(model, train, test)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
