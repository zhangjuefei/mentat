from collections import OrderedDict
import copy

from .base import MultiTrainer


class MultiModelTrainer(MultiTrainer):
    def __init__(self, models, train_fraction=0.75, evaluator=None, metric=None):
        super().__init__(train_fraction, evaluator, metric)

        self.models = OrderedDict(models)

    def train(self, data):
        train, test = data.split(self.train_fraction)

        self.train_models(self.models.values(), train)

        # evaluate
        if self.metric:
            for name, model in self.models.items():
                evaluator = copy.deepcopy(self.evaluator)
                self.evaluators[name] = evaluator.fit(model.evaluate(test))
                self.metrics[name] = self.evaluators[name][self.metric]

            self.best_model = self.models[max(self.metrics, key=self.metrics.get)]
            self.evaluator = self.evaluators[max(self.metrics, key=self.metrics.get)]

    def get_evaluator(self):
        return self.evaluator
