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
print("Accuracies of 3 models:")
for name, accuracy in pipeline.get_operator("trainer").metrics.items():
    print("model: {:s}  accuracy: {:.6f}".format(name, accuracy))

# metrics of the chosen(best) DNN
eva = pipeline.get_operator("trainer").get_evaluator()
print("\nconfusion matrix of the best model:")
print(eva["confusion_matrix"])
print("\nclassification report of the best model:")
print(eva["classification_report"])
print("\naccuracy of the best model")
print(eva["accuracy"])

#  use pipeline to predict
predict = pipeline.evaluate(to_be_predicted)

# ZDataFrame is callable, return the data(pandas DataFrame) it contains
print("\nsome prediction examples:")
print(predict().head(5))
```

##### output

```
C:\Users\chaos\AppData\Local\Programs\Python\Python36-32\python.exe D:/projects/mentat/mentat/test/multi_trainer_test.py
Accuracies of 3 models:
model: dnn_1  accuracy: 1.000000
model: dnn_2  accuracy: 0.967742
model: dnn_3  accuracy: 1.000000

confusion matrix of the best model:
predict          Iris-setosa  Iris-versicolor  Iris-virginica  total
true
Iris-setosa               13                0               0     13
Iris-versicolor            0               11               0     11
Iris-virginica             0                0               7      7
total                     13               11               7     31

classification report of the best model:
                 precision  recall  f1 score
Iris-setosa            1.0     1.0       1.0
Iris-versicolor        1.0     1.0       1.0
Iris-virginica         1.0     1.0       1.0

accuracy of the best model
1.0

some prediction examples:
  predict_category   Iris-setosa  Iris-versicolor  Iris-virginica   Id  \
0   Iris-virginica  5.132001e-16         0.011557        0.988443  132
1   Iris-virginica  5.143205e-28         0.000007        0.999993  106
2   Iris-virginica  2.132122e-19         0.002446        0.997554  143
3  Iris-versicolor  1.524778e-03         0.997197        0.001278   97
4   Iris-virginica  7.390647e-25         0.000054        0.999946  129

   SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm          Species
0       2.569442      1.652052       1.552178      1.103227   Iris-virginica
1       2.199527     -0.205677       1.664075      1.232367   Iris-virginica
2      -0.019964     -0.902326       0.824844      0.974086   Iris-virginica
3      -0.143269     -0.437893       0.321306      0.199245  Iris-versicolor
4       0.719866     -0.670109       1.104588      1.232367   Iris-virginica

Process finished with exit code 0

```
