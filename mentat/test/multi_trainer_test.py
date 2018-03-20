from mentat import ZDataFrame
from mentat.preprocessor import StandardScaler
from mentat.model import LogisticRegression, DNN
from mentat.trainer import MultiModelTrainer
from mentat.pipeline import Pipeline
from mentat.evaluator import ClassificationEvaluator
import pandas as pd
import numpy as np

# load and construct the data frame
df = pd.read_csv("../data/bird.csv")
data = ZDataFrame(df, response_column="type", ignores=["id"], response_encode="multiclass").impute("mean")

# 3 neural networks.
classifiers = {
    "dnn_1": DNN(input_shape=len(data.feature_cols), shape=[10, len(data.category)],
                 activations=["sigmoid", "identity"], eta=1.5, softmax=True, max_epochs=50,
                 regularization=0,
                 minibatch_size=20, momentum=0.9, decay_power=0.2, verbose=True
                 ),
    "dnn_2": DNN(input_shape=len(data.feature_cols), shape=[20, len(data.category)],
                 activations=["sigmoid", "identity"], eta=1.5, softmax=True, max_epochs=50,
                 regularization=0,
                 minibatch_size=20, momentum=0.9, decay_power=0.2, verbose=True
                 ),
    "dnn_3": DNN(input_shape=len(data.feature_cols), shape=[60, len(data.category)],
                 activations=["sigmoid", "identity"], eta=1.5, softmax=True, max_epochs=50,
                 regularization=0,
                 minibatch_size=20, momentum=0.9, decay_power=0.2, verbose=True
                 ),
}

# pipeline
pipeline = Pipeline(
    {
        "preprocessor": StandardScaler(),  # preprocessor: standard scaler
        "trainer": MultiModelTrainer(classifiers, train_fraction=0.7, evaluator=ClassificationEvaluator(),
                                     metric="accuracy"),  # trivial trainer
    }
)

# train
pipeline.fit(data)

print("\n---------- Models' Accuracy -----------")
for name, accuracy in pipeline.get_operator("trainer").metrics.items():
    print("model: {:s}  accuracy: {:.6f}".format(name, accuracy))

# predict
eva = pipeline.get_operator("trainer").get_evaluator()

# classification evaluations
print("\n---------- Confusion Matrix ----------")
print(eva.confusion_matrix())
print("\n------- Classification Report --------")
print(eva.report())
print("\n------------- Accuracy ---------------")
print(eva.accuracy())
