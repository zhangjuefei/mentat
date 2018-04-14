import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles

from mentat import ZDataFrame
from mentat.evaluator import ClassificationEvaluator
from mentat.model import DNN
from mentat.preprocessor import StandardScaler

max_epochs = 40
hidden_neurons = 3

cm = plt.cm.coolwarm
cm_bright = ListedColormap(["#0000FF", "#FF0000"])

# figure axes
axes = []
fig = plt.figure(figsize=(8, 8))
axes.append(fig.add_subplot(2, 2, 1, projection="3d"))
axes.append(fig.add_subplot(2, 2, 2))
axes.append(fig.add_subplot(2, 2, 3, projection="3d"))
axes.append(fig.add_subplot(2, 2, 4))
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)

# data points
X, y = make_circles(n_samples=120, noise=0.24, factor=0.2, random_state=42)
points = StandardScaler().fit_evaluate(
    ZDataFrame(pd.DataFrame(np.c_[X, y], columns=["x", "y", "z"]), response_column="z",
               response_encode="multiclass").impute("mean"))

x_min = points().describe().loc["min", "x"] - 0.5
x_max = points().describe().loc["max", "x"] + 0.5
y_min = points().describe().loc["min", "y"] - 0.5
y_max = points().describe().loc["max", "y"] + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

train, test = points.split(0.7)

# neural network
dnn = DNN(input_shape=2, shape=[hidden_neurons, 2], activations=["sigmoid", "identity"], eta=0.2, softmax=True,
          max_epochs=1, minibatch_size=20, verbose=True, decay_power=0.2, regularization=1e-4)
evaluator = ClassificationEvaluator()

# loss
loss = []
train_accuracy = []
test_accuracy = []


def draw(idx):
    print("epochs: {:d}".format(idx))
    bottom = -.5
    c_train = 1 - train()["z"]
    c_test = 1 - test()["z"]

    dnn.fit(train)
    train_predict = dnn.evaluate(train)
    hidden_outputs_train = dnn.outputs[dnn.depth - 1].transpose().A
    test_predict = dnn.evaluate(test)
    hidden_outputs_test = dnn.outputs[dnn.depth - 1].transpose().A

    train_accuracy.append(evaluator.fit(train_predict).metrics["accuracy"])
    test_accuracy.append(evaluator.fit(test_predict).metrics["accuracy"])
    loss.append(dnn.epoch_loss[-1])

    if idx == 0:
        train_accuracy.append(train_accuracy[0])
        test_accuracy.append(test_accuracy[0])
        loss.append(loss[0])

    probability = dnn.predict(np.c_[xx.ravel(), yy.ravel()])[:, 0].reshape(xx.shape)

    train_acc = train_accuracy[-1] if len(train_accuracy) else 0.0
    test_acc = test_accuracy[-1] if len(test_accuracy) else 0.0
    current_loss = loss[-1] if len(loss) else 0.0

    axes[0].clear()
    axes[0].set_title(r"$probability\ surface$", fontsize=8)
    axes[0].plot_surface(xx, yy, probability, rstride=1, cstride=1, alpha=0.6, cmap=cm)
    axes[0].contourf(xx, yy, probability, zdir='z', offset=bottom, alpha=0.6, cmap=cm)
    axes[0].scatter(train()["x"], train()["y"], bottom, c=c_train, cmap=cm_bright, edgecolors='k',
                    s=10)
    axes[0].scatter(test()["x"], test()["y"], bottom, c=c_test, cmap=cm_bright, edgecolors='k',
                    alpha=0.5, s=10)
    axes[0].set_xlim(xx.min(), xx.max())
    axes[0].set_ylim(yy.min(), yy.max())
    axes[0].set_xlabel(r"$x_1$", fontsize=8)
    axes[0].set_ylabel(r"$x_2$", fontsize=8)
    axes[0].tick_params(labelsize=8)
    axes[0].view_init(40, -45)

    axes[1].clear()
    axes[1].set_title(r"$epochs: {:d}/{:d}$".format(idx, max_epochs), fontsize=8)
    axes[1].scatter(train()["x"], train()["y"], c=c_train, cmap=cm_bright, edgecolors='k', s=10)
    axes[1].scatter(test()["x"], test()["y"], c=c_test, cmap=cm_bright, edgecolors='k', alpha=0.5,
                    s=10)
    axes[1].contourf(xx, yy, probability, cmap=cm, alpha=.6)
    axes[1].set_xlim(xx.min(), xx.max())
    axes[1].set_ylim(yy.min(), yy.max())
    axes[1].set_xlabel(r"$x_1$", fontsize=8)
    axes[1].set_ylabel(r"$x_2$", fontsize=8)
    axes[1].tick_params(labelsize=8)
    axes[1].grid()

    axes[2].clear()
    axes[2].set_title(r"$outputs\ of\ 3\ hidden\ neurons$", fontsize=8)
    axes[2].scatter(hidden_outputs_train[:, 0], hidden_outputs_train[:, 1], hidden_outputs_train[:, 2],
                    c=c_train,
                    cmap=cm_bright, edgecolors="k", s=10)
    axes[2].scatter(hidden_outputs_test[:, 0], hidden_outputs_test[:, 1], hidden_outputs_test[:, 2],
                    c=c_test,
                    cmap=cm_bright, edgecolors="k", alpha=0.6, s=10)
    axes[2].set_xlabel(r"$1st\ neuron$", fontsize=8)
    axes[2].set_ylabel(r"$2nd\ neuron$", fontsize=8)
    axes[2].set_zlabel(r"$3rd\ neuron$", fontsize=8)
    axes[2].tick_params(labelsize=8)
    axes[2].view_init(40, 45)
    axes[2].grid()

    axes[3].clear()
    axes[3].set_title(r"$loss\ and\ metrics$", fontsize=8)
    axes[3].plot(loss, linewidth=0.5)
    axes[3].fill_between(np.arange(len(loss)), [0] * len(loss), loss, alpha=0.6)
    axes[3].plot(train_accuracy, linewidth=1)
    axes[3].plot(test_accuracy, linewidth=1)
    axes[3].set_ylim([0, 1])
    axes[3].set_xlim([0, len(loss)-1 if len(loss) else 0])
    axes[3].legend(["loss ({:.3f})".format(current_loss), "accuracy train ({:.3f})".format(train_acc),
                    "accuracy test ({:.3f})".format(test_acc)], loc="lower left", fontsize=8)
    axes[3].set_xlabel(r"$epochs$", fontsize=8)
    axes[3].tick_params(labelsize=8)
    axes[3].grid()


# init
def init():
    draw(0)


# update
def update(idx):
    draw(idx)


anim = animation.FuncAnimation(fig, update, init_func=init, frames=max_epochs, interval=220, blit=False)
anim.save('dnn_animation.gif', writer='imagemagick')
