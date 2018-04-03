from collections import OrderedDict
import itertools
import copy

from .base import MultiTrainer
from ..exception import ParameterException


class GridSearchTrainer(MultiTrainer):
    def __init__(self, model, params, train_fraction=0.75, evaluator=None, metric=None):
        super().__init__(train_fraction, evaluator, metric)
        self.train_fraction = train_fraction
        self.model = model
        self.models = OrderedDict()
        self.params = OrderedDict(params)

    def train(self, data):
        train, test = data.split(self.train_fraction)

        for param_group in self.get_grid_params():
            name = "model_" + "#".join(list(map(lambda t: str(t[0]) + "_" + str(t[1]), list(param_group))))
            model = copy.deepcopy(self.model)
            for param, value in param_group:

                if not hasattr(model, param):
                    raise ParameterException("no such param: {:s}".format(param))

                setattr(model, param, value)
            self.models[name] = model

        self.train_models(self.models.values(), train)

        # evaluate
        if self.metric:
            for name, model in self.models.items():
                evaluator = copy.deepcopy(self.evaluator)
                self.evaluators[name] = evaluator.fit(model.evaluate(test))
                self.metrics[name] = self.evaluators[name][self.metric]

            self.best_model = self.models[max(self.metrics, key=self.metrics.get)]
            self.evaluator = self.evaluators[max(self.metrics, key=self.metrics.get)]

    def get_grid_params(self):
        param_arr = []
        for param, values in self.params.items():
            pv = []
            for value in values:
                pv.append((param, value))
            param_arr.append(pv)

        return list(itertools.product(*param_arr))

    def get_evaluator(self):
        return self.evaluator
