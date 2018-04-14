MENTAT
==
<pre>
The future will be an ever more demanding struggle against the limitations of our intelligence,
not a comfortable hammock in which we can lie down to be waited upon by our robot slaves.

                                                -- Norbert Wiener “The Human Use Of Human Beings”
</pre>

#####[Graph illustrating the training of a ANN](https://raw.githubusercontent.com/zhangjuefei/mentat/master/mentat/test/dnn_animation.py)
</br>

![animation](https://raw.githubusercontent.com/zhangjuefei/mentat/master/mentat/test/pic/dnn_animation.gif)
</br>

#####Mentat Usage Example:
</br>

```python
import pandas as pd

from mentat import ZDataFrame
from mentat.evaluator import ClassificationEvaluator
from mentat.model import DNN
from mentat.pipeline import Pipeline
from mentat.preprocessor import StandardScaler
from mentat.trainer import MultiModelTrainer

# load and construct the data frame
df = pd.read_csv("../data/bird.csv")
data = ZDataFrame(df, response_column="type", ignores=["id"], response_encode="multiclass").impute("mean")

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

# the accuracies of 3 DNN
for name, accuracy in pipeline.get_operator("trainer").metrics.items():
    print("model: {:s}  accuracy: {:.6f}".format(name, accuracy))

# metrics of the chosen(best) DNN
eva = pipeline.get_operator("trainer").get_evaluator()
print(eva.confusion_matrix())
print(eva.report())
print(eva.accuracy())

#  use pipeline to predict
predict = pipeline.evaluate(to_be_predicted)

# ZDataFrame is callable, return the data(pandas DataFrame) it contains
print(predict().head(5))
```
