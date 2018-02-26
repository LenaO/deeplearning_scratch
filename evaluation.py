from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import itertools
from tabulate import tabulate
import math
import test_params as test

def coords_xyz_plot(y_true, y_pred, save_file=None, shape=(1000,1000), dpi=100):
    print(y_true.shape)
    print(y_pred.shape)
    fig, axes = plt.subplots(1,3,figsize=(shape[0]/float(dpi), shape[1]/float(dpi)), dpi=dpi)
    for i in range(3):
        ax = axes[i]
        yt = y_true[:,i].flatten()
        yp = y_pred[:,i].flatten()
        ax.plot(yt, yp, 'o', markersize=2)
    axes[0].set_ylabel('Predicted')
    axes[1].set_xlabel('True')
    plt.tight_layout()
    if save_file is not None:
        fig.savefig(save_file, format='png')
    return fig


def coords_xyz_plot_for_tensorboard(y_true, y_pred, shape):
    fig = coords_xyz_plot(y_true, y_pred, shape=(shape[2], shape[1]))
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    img = np.fromstring(buf, dtype=np.uint8).reshape(*shape)
    return img

def distance_plot(y_true, y_pred, save_file=None, shape=(1000,1000)):
    axlim = [test.min_distance, const.max_distance]
    fig = plt.figure(figsize=(shape[0]/300., shape[1]/300.), dpi=300)
    ax = fig.add_subplot(111, ylim=axlim, xlim=axlim)
    ax.plot(y_true, y_pred, 'o', markersize=2)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    if save_file is not None:
        fig.savefig(save_file, format='png')
    return fig


def distance_plot_for_tensorboard(y_true, y_pred, shape):
    fig = distance_plot(y_true, y_pred, shape=shape[1:3])
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    img = np.fromstring(buf, dtype=np.uint8).reshape(*shape)
    return img


def confusion_matrix_plot(y_true, y_pred, labels, normalize=False, cmap=plt.cm.Blues, shape=(1000,1000), text=True, ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = metrics.confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    dpi = None
    if shape is not None:
        shape = (shape[0]/300., shape[1]/300.)
        dpi = 300
    if ax is None:
        fig = plt.figure(figsize=shape, dpi=dpi)
        ax = fig.add_subplot(111)

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    if text:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    return ax.figure


def confusion_matrix_for_tensorboard(y_true, y_pred, labels, shape):
    fig = confusion_matrix_plot(y_true, y_pred, labels, normalize=True, shape=shape[1:3])
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    img = np.fromstring(buf, dtype=np.uint8).reshape(*shape)
    return img

