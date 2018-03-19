from .base import Trainer


class TrivialTrainer(Trainer):
    def __init__(self, model, train_fraction=0.75, evaluator=None):
        Trainer.__init__(self, model)
        self.train_fraction = train_fraction
        self.best_model = model
        self.evaluator = evaluator

    def train(self, data):
        train, test = data.split(self.train_fraction)
        self.best_model.fit(train)

        if self.evaluator:
            self.evaluator.fit(self.best_model.evaluate(test))

    def get_evaluator(self):
        return self.evaluator
