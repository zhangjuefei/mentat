import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles ,make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mentat.model import DNN

max_epochs = 30
cm = plt.cm.coolwarm
cm_bright = ListedColormap(["#0000FF", "#FF0000"])
plt.tight_layout()

axes = []
fig = plt.figure(figsize=(8, 8))
axes.append(fig.add_subplot(2, 2, 1, projection="3d"))
axes.append(fig.add_subplot(2, 2, 2))
axes.append(fig.add_subplot(2, 2, 3, projection="3d"))
axes.append(fig.add_subplot(2, 2, 4))

# data points and label
X, y = make_circles(n_samples=120, noise=0.24, factor=0.2, random_state=42)
y = np.c_[[y, 1 - y]].T
X = StandardScaler().fit_transform(X)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=42)
train_label, test_label = np.argmax(y_train, axis=1), np.argmax(y_test, axis=1)

# neural network
hidden_neurons = 3
dnn = DNN(input_shape=2, shape=[hidden_neurons, 2], activations=["sigmoid", "identity"], eta=0.2, softmax=True,
          max_epochs=1,
          minibatch_size=20, verbose=True, decay_power=0.2, regularization=1e-4)

# loss
loss = []
train_accuracy = []
test_accuracy = []


def draw(probability, idx):
    print("epochs: {:d}".format(idx))
    bottom = -.5
    train_acc = train_accuracy[-1] if len(train_accuracy) else 0.0
    test_acc = test_accuracy[-1] if len(test_accuracy) else 0.0
    current_loss = loss[-1] if len(loss) else 0.0

    axes[0].clear()
    axes[0].set_title(r"$probability\ surface$", fontsize=8)
    axes[0].plot_surface(xx, yy, probability, rstride=1, cstride=1, alpha=0.6, cmap=cm)
    axes[0].contourf(xx, yy, probability, zdir='z', offset=bottom, alpha=0.6, cmap=cm)
    axes[0].scatter(X_train[:, 0], X_train[:, 1], bottom, c=y_train[:, 0], cmap=cm_bright, edgecolors='k', s=10)
    axes[0].scatter(X_test[:, 0], X_test[:, 1], bottom, c=y_test[:, 0], cmap=cm_bright, edgecolors='k', alpha=0.5, s=10)
    axes[0].set_xlim(xx.min(), xx.max())
    axes[0].set_ylim(yy.min(), yy.max())
    # axes[0].set_zlim(bottom, 1.2)
    axes[0].set_xlabel(r"$x_1$", fontsize=8)
    axes[0].set_ylabel(r"$x_2$", fontsize=8)
    axes[0].tick_params(labelsize=8)
    axes[0].view_init(40, -45)

    axes[1].clear()
    axes[1].set_title(r"$epochs: {:d}/{:d}$".format(idx, max_epochs), fontsize=8)
    axes[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train[:, 0], cmap=cm_bright, edgecolors="k", s=10)
    axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test[:, 0], cmap=cm_bright, edgecolors="k", alpha=0.6, s=10)
    axes[1].contourf(xx, yy, probability, cmap=cm, alpha=.6)
    axes[1].set_xlim(xx.min(), xx.max())
    axes[1].set_ylim(yy.min(), yy.max())
    axes[1].set_xlabel(r"$x_1$", fontsize=8)
    axes[1].set_ylabel(r"$x_2$", fontsize=8)
    axes[1].tick_params(labelsize=8)
    axes[1].grid()

    axes[3].clear()
    axes[3].set_title(r"$loss\ and\ metrics$", fontsize=8)
    axes[3].plot(loss, linewidth=0.5)
    axes[3].fill_between(np.arange(len(loss)), [0] * len(loss), loss, alpha=0.6)
    axes[3].plot(train_accuracy, linewidth=1)
    axes[3].plot(test_accuracy, linewidth=1)
    axes[3].set_ylim([0, 1])
    axes[3].set_xlim([0, len(loss) if len(loss) else 1])
    axes[3].legend(["loss ({:.3f})".format(current_loss), "accuracy train ({:.3f})".format(train_acc),
                    "accuracy test ({:.3f})".format(test_acc)], loc="lower left", fontsize=8)
    axes[3].set_xlabel(r"$epochs$", fontsize=8)
    axes[3].tick_params(labelsize=8)
    axes[3].grid()

    axes[2].clear()
    axes[2].set_title(r"$outputs\ of\ 3\ hidden\ neurons$", fontsize=8)
    dnn.predict(X_train)
    transformed_train = dnn.outputs[dnn.depth - 1].transpose().A
    dnn.predict(X_test)
    transformed_test = dnn.outputs[dnn.depth - 1].transpose().A
    axes[2].scatter(transformed_train[:, 0], transformed_train[:, 1], transformed_train[:, 2], c=y_train[:, 0],
                    cmap=cm_bright, edgecolors="k", s=10)
    axes[2].scatter(transformed_test[:, 0], transformed_test[:, 1], transformed_test[:, 2], c=y_test[:, 0],
                    cmap=cm_bright, edgecolors="k", alpha=0.6, s=10)
    axes[2].set_xlabel(r"$1st\ neuron$", fontsize=8)
    axes[2].set_ylabel(r"$2nd\ neuron$", fontsize=8)
    axes[2].set_zlabel(r"$3rd\ neuron$", fontsize=8)
    axes[2].tick_params(labelsize=8)
    axes[2].view_init(40, 45)
    axes[2].grid()


# draw the in
probability = dnn.predict(np.c_[xx.ravel(), yy.ravel()])[:, 0].reshape(xx.shape)
draw(probability, 0)


# init
def init():
    pass


# update
def update(i):
    dnn.train(X_train, y_train)
    probability = dnn.predict(np.c_[xx.ravel(), yy.ravel()])[:, 0].reshape(xx.shape)

    train_accuracy.append(accuracy_score(train_label, np.argmax(dnn.predict(X_train), axis=1)))
    test_accuracy.append(accuracy_score(test_label, np.argmax(dnn.predict(X_test), axis=1)))
    loss.append(dnn.epoch_loss[-1])
    draw(probability, i)


anim = animation.FuncAnimation(fig, update, init_func=init, frames=max_epochs, interval=220, blit=False)

# anim.save('dnn_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# plt.rcParams['animation.convert_path'] = 'C:\Program Files\ImageMagick-6.9.0-Q16\convert.exe'
anim.save('dnn_animation.gif', writer='imagemagick')
# anim.save('dnn_animation_html/dnn_animation.html')

# plt.show()
