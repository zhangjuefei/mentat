from mentat import ZDataFrame
from mentat.preprocessor import StandardScaler
from mentat.model import DNN
from mentat.trainer import TrivialTrainer
from mentat.pipeline import Pipeline
from mentat.evaluator import ClassificationEvaluator
import pandas as pd

# load and construct the data frame
bird = pd.read_csv("../data/bird.csv")
data = ZDataFrame(bird, response_column="type", ignores=["id"], response_encode="multiclass").impute("mean")

# keep a test set out
train, test = data.split(0.6)

# deep neural network
dnn = DNN(
    input_shape=len(data.feature_cols),
    shape=[20, len(data.category)],
    activations=["sigmoid", "identity"],
    eta=1.0,
    threshold=1e-5,
    softmax=True,
    max_epochs=20,
    regularization=0.0001,
    minibatch_size=10,
    momentum=0.9,
    decay_power=0.2,
    verbose=True
)

# pipeline
pipeline = Pipeline(
    {
        "preprocessor": StandardScaler(),  # preprocessor: standard scaler
        "trainer": TrivialTrainer(dnn, train_fraction=0.7)  # trivial trainer
    }
)

# train
pipeline.fit(train)

# predict
result = pipeline.evaluate(test)

# confusion matrix
eva = ClassificationEvaluator()
eva.fit(result)
print(eva.confusion_matrix())
print("accuracy: {:.3f}".format(eva.accuracy()))
