import pandas as pd
import unittest
import os
import numpy as np

from mentat import ZDataFrame
from mentat.evaluator import ClassificationEvaluator
from mentat.model import DNN, LogisticRegression, LinearRegression
from mentat.pipeline import Pipeline
from mentat.preprocessor import StandardScaler
from mentat.trainer import MultiModelTrainer

DATA_PATH = "../data"

class ModelTest(unittest.TestCase):

    def setUp(self):

        self.bird = ZDataFrame(
            pd.read_csv(DATA_PATH + os.path.sep + "bird.csv"),
            "type",
            ["id"],
            "multiclass"
        ).impute("mean")

        self.pelvis = ZDataFrame(
            pd.read_csv(DATA_PATH + os.path.sep + "column_2C_weka.csv"),
            "class",
            [],
            "binary"
        ).impute("mean")

        x = np.arange(-2 * np.pi, 2 * np.pi, .1)
        true_y = 2. * 3. * x
        y = true_y + np.random.normal(loc=0., scale=0.3, size=len(x))
        self.regression_data = ZDataFrame(
            pd.DataFrame({"x": x, "y": y}),
            "y",
        )

    def test_multi_trainer(self):
        # load and construct the data frame
        data = self.bird

        # number of features(input size) and number of categories(output size)
        input_size = len(data.feature_cols)
        output_size = len(data.category)

        # split the data into train(and test) data set and data set to be predicted
        train_and_test, to_be_predicted = data.split(.7)

        # construct 3 models(DNN) with different hyper-parameters(size of hidden layer and max epochs here)
        dnns = {
            "dnn_1": DNN(input_size, [2, output_size], ["relu", "identity"], softmax=True, max_epochs=2),
            "dnn_2": DNN(input_size, [20, output_size], ["relu", "identity"], softmax=True, max_epochs=20),
            "dnn_3": DNN(input_size, [60, output_size], ["relu", "identity"], softmax=True, max_epochs=30)
        }

        # construct a pipeline contains a standard scaler and a multi-model trainer(train 3 DNN parallel)
        pipeline = Pipeline(
            {
                "preprocessor": StandardScaler(),
                "trainer": MultiModelTrainer(dnns, train_fraction=.7, evaluator=ClassificationEvaluator(),
                                             metric="accuracy")
            }
        )

        # fit the pipeline
        pipeline.fit(train_and_test)

        # metrics of the chosen(best) DNN
        eva = pipeline.get_operator("trainer").get_evaluator()
        self.assertGreater(eva.accuracy(), 0.75, "accuracy is too low")

    def test_linear_regression(self):

        lr_arr = {
            "sgd_lr": LinearRegression(method="sgd", eta=0.01, decay_power=0.5, regularization=3.0, max_epochs=1000,
                             minibatch_size=100),
            "lr": LinearRegression(method="analytic", regularization=0)
        }

        pipeline = Pipeline(
            {
                "trainer": MultiModelTrainer(lr_arr, train_fraction=.7)
            }
        )

        pipeline.fit(self.regression_data)


if __name__ == "__main__":
    unittest.main()