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
print("\nAccuracies of 3 Models:\n")
for name, accuracy in pipeline.get_operator("trainer").metrics.items():
    print("model: {:s}  accuracy: {:.6f}".format(name, accuracy))

# metrics of the chosen(best) DNN
eva = pipeline.get_operator("trainer").get_evaluator()
print("\nConfusion Matrix of the Best Model:\n")
print(eva["confusion_matrix"])
print("\nClassification Report of the Best Model:\n")
print(eva["classification_report"])
print("\nAccuracy of the Best Model:\n")
print(eva["accuracy"])

#  use pipeline to predict
predict = pipeline.evaluate(to_be_predicted)

# ZDataFrame is callable, return the data(pandas DataFrame) it contains
print("\nSome Prediction Examples:\n")
print(predict().head(5))
```

##### output

```
Accuracies of 3 Models:

model: dnn_1  accuracy: 0.935484
model: dnn_2  accuracy: 0.903226
model: dnn_3  accuracy: 0.870968

Confusion Matrix of the Best Model:

predict          Iris-setosa  Iris-versicolor  Iris-virginica  total
true                                                                
Iris-setosa               12                0               0     12
Iris-versicolor            0               12               0     12
Iris-virginica             0                2               5      7
total                     12               14               5     31

Classification Report of the Best Model:

                 precision    recall  f1 score
Iris-setosa       1.000000  1.000000  1.000000
Iris-versicolor   0.857143  1.000000  0.795918
Iris-virginica    1.000000  0.714286  0.612245

Accuracy of the Best Model:

0.9354838709677419

Some Prediction Examples:

  predict_category   Iris-setosa  Iris-versicolor  Iris-virginica   Id  \
0      Iris-setosa  9.917974e-01         0.008183        0.000020   13   
1      Iris-setosa  9.917974e-01         0.008183        0.000020   26   
2  Iris-versicolor  1.116974e-07         0.517066        0.482934  128   
3      Iris-setosa  9.917974e-01         0.008183        0.000020   29   
4   Iris-virginica  6.104579e-13         0.000141        0.999859  130   

   SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm         Species  
0      -1.264247     -0.125569      -1.317562     -1.428032     Iris-setosa  
1      -1.025065     -0.125569      -1.204904     -1.296618     Iris-setosa  
2       0.290435     -0.125569       0.653953      0.806006  Iris-virginica  
3      -0.785883      0.816201      -1.317562     -1.296618     Iris-setosa  
4       1.605935     -0.125569       1.160914      0.543178  Iris-virginica
```
