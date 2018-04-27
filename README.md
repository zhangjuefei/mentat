MENTAT
==
<pre>
The future will be an ever more demanding struggle against the limitations of our intelligence,
not a comfortable hammock in which we can lie down to be waited upon by our robot slaves.

                                                -- Norbert Wiener “The Human Use Of Human Beings”
</pre>
</br>

#### [Illustration of the training process of a artificial neural network](https://github.com/zhangjuefei/mentat/blob/master/mentat/test/dnn_animation.py)


![animation](https://raw.githubusercontent.com/zhangjuefei/mentat/master/mentat/test/pic/dnn_animation.gif)



#### Mentat Usage Example:



```python
import pandas as pd

from mentat import ZDataFrame
from mentat.evaluator import ClassificationEvaluator
from mentat.model import DNN
from mentat.pipeline import Pipeline
from mentat.preprocessor import StandardScaler
from mentat.trainer import MultiModelTrainer

# load and construct the data frame
df = pd.read_csv("../data/Iris.csv")
data = ZDataFrame(df, response_column="Species", ignores=["Id"], response_encode="multiclass").impute("mean")

# number of categories(output size)
output_size = len(data.category)

# split the data into train(and test) data set and data set to be predicted
train_and_test, to_be_predicted = data.split(.7)

# construct 3 models(DNN) with different hyper-parameters(size of hidden layer and max epochs here)
dnns = {
    "dnn_1": DNN([2, output_size], ["relu", "identity"], softmax=True, max_epochs=2),
    "dnn_2": DNN([20, output_size], ["relu", "identity"], softmax=True, max_epochs=20),
    "dnn_3": DNN([60, output_size], ["relu", "identity"], softmax=True, max_epochs=30)
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
print(eva["confusion_matrix"])
print(eva["classification_report"])
print(eva["accuracy"])

#  use pipeline to predict
predict = pipeline.evaluate(to_be_predicted)

# ZDataFrame is callable, return the data(pandas DataFrame) it contains
print(predict().head(5))
```

####output

```
model: dnn_1  accuracy: 0.967742
model: dnn_2  accuracy: 0.903226
model: dnn_3  accuracy: 0.903226

predict          Iris-setosa  Iris-versicolor  Iris-virginica  total
true
Iris-setosa               15                0               0     15
Iris-versicolor            0                8               0      8
Iris-virginica             0                1               7      8
total                     15                9               7     31

                 precision  recall  f1 score
Iris-setosa       1.000000   1.000  1.000000
Iris-versicolor   0.888889   1.000  0.839506
Iris-virginica    1.000000   0.875  0.820312
0.967741935484

  predict_category   Iris-setosa  Iris-versicolor  Iris-virginica   Id  \
0  Iris-versicolor  5.811692e-03         0.991297        0.002891   77
1      Iris-setosa  9.754901e-01         0.024505        0.000005   32
2   Iris-virginica  1.457382e-12         0.099564        0.900436  147
3   Iris-virginica  1.222260e-13         0.000110        0.999890  142
4      Iris-setosa  9.754901e-01         0.024505        0.000005    6

   SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm          Species
0       1.115819     -0.630728       0.585941      0.285610  Iris-versicolor
1      -0.508195      0.711700      -1.225196     -1.001475      Iris-setosa
2       0.535814     -1.301942       0.695707      0.929153   Iris-virginica
3       1.231820      0.040486       0.750589      1.443987   Iris-virginica
4      -0.508195      1.830390      -1.115430     -1.001475      Iris-setosa
```
