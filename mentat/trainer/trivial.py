from .base import Trainer


class TrivialTrainer(Trainer):
    def __init__(self, model, train_fraction=0.75):
        Trainer.__init__(self, model)
        self.train_fraction = train_fraction
        self.best_model = model

    def train(self, data):
        train, test = data.split(self.train_fraction)
        self.best_model.fit(train)
        test_result = self.best_model.evaluate(test)
