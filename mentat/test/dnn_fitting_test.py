import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mentat.model import DNN
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

hidden_layer_size = [3, 5, 8]  # 隐藏层神经元个数（所有隐藏层都取同样数量神经元）
hidden_layers = 1  # 隐藏层数量
hidden_layer_activation_func = "sigmoid"  # 隐藏层激活函数
learning_rate = 0.6  # 学习率
max_epochs = 300  # 训练 epoch 数量
regularization_strength = 0  # 正则化强度
minibatch_size = 40  # mini batch 样本数
momentum = 0.7  # 冲量惯性
decay_power = 0.3  # 学习率衰减指数


def f1(x):
    return (x[:, 0] ** 2 + x[:, 1] ** 2).reshape((len(x), 1))


def f2(x):
    return (x[:, 0] ** 2 - x[:, 1] ** 2).reshape((len(x), 1))


def f3(x):
    return (np.cos(1.2 * x[:, 0]) * np.cos(1.2 * x[:, 1])).reshape((len(x), 1))


funcs = [f1, f2, f3]

X = np.random.uniform(low=-2.0, high=2.0, size=(100, 2))
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

# 模型
names = ["{:d} neurons per layer".format(hs) for hs
         in hidden_layer_size]

classifiers = [
    DNN(input_shape=2, shape=[hs] * hidden_layers + [1],
        activations=[hidden_layer_activation_func] * hidden_layers + ["identity"], eta=learning_rate, threshold=0.001,
        softmax=False, max_epochs=max_epochs, regularization=regularization_strength, verbose=True,
        minibatch_size=minibatch_size, momentum=momentum, decay_power=decay_power) for hs in
    hidden_layer_size
]

figure = plt.figure(figsize=(5 * len(classifiers) + 2, 4 * len(funcs)))
cm = plt.cm.PuOr
cm_bright = ListedColormap(["#DB9019", "#00343F"])
i = 1

for cnt, f in enumerate(funcs):

    zz = f(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    z = f(X)

    ax = figure.add_subplot(len(funcs), len(classifiers) + 1, i, projection="3d")

    if cnt == 0:
        ax.set_title("data")

    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, alpha=0.6, cmap=cm)
    ax.contourf(xx, yy, zz, zdir='z', offset=zz.min(), alpha=0.6, cmap=cm)
    ax.scatter(X[:, 0], X[:, 1], z.ravel(), cmap=cm_bright, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_zlim(zz.min(), zz.max())

    i += 1

    for name, clf in zip(names, classifiers):

        print("model: {:s} training.".format(name))

        ax = plt.subplot(len(funcs), len(classifiers) + 1, i)
        clf.train(X, z)
        predict = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax = figure.add_subplot(len(funcs), len(classifiers) + 1, i, projection="3d")

        if cnt == 0:
            ax.set_title(name)

        ax.plot_surface(xx, yy, predict, rstride=1, cstride=1, alpha=0.6, cmap=cm)
        ax.contourf(xx, yy, predict, zdir='z', offset=zz.min(), alpha=0.6, cmap=cm)
        ax.scatter(X[:, 0], X[:, 1], z.ravel(), cmap=cm_bright, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_zlim(zz.min(), zz.max())

        i += 1
        print("model: {:s} train finished.".format(name))

plt.tight_layout()
plt.savefig(
    "dnn_fitting_{:d}_{:.6f}_{:d}_{:.6f}_{:.3f}_{:3f}.png".format(hidden_layers, learning_rate, max_epochs,
                                                                      regularization_strength, momentum, decay_power))