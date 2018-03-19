MENTAT
==
A machine learning library build on python, pandas and numpy
</br>

```python
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

# deep neural network
dnn = DNN(
    input_shape=len(data.feature_cols),
    shape=[10, 10, len(data.category)],
    activations=["sigmoid", "sigmoid", "identity"],
    eta=1.5,
    threshold=1e-5,
    softmax=True,
    max_epochs=50,
    regularization=0.0001,
    minibatch_size=10,
    momentum=0.9,
    decay_power=0.2,
    verbose=True
)

# pipeline
pipeline = Pipeline(
    {
        # preprocessor: standard scaler
        "preprocessor": StandardScaler(),
        # trivial trainer (train and test)
        "trainer": TrivialTrainer(dnn, train_fraction=0.7, evaluator=ClassificationEvaluator()),
    }
)

# train
pipeline.fit(data)

# predict
predict_result = pipeline.evaluate(data)

# classification evaluations
eva = pipeline.get_operator("trainer").get_evaluator()
print("\n---------- Confusion Matrix ----------")
print(eva.confusion_matrix())
print("\n------- Classification Report --------")
print(eva.report())
```