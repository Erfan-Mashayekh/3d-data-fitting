import matplotlib.pyplot as plt
import numpy as np

from utilities import *


def plot_loss(history):
    plt.plot(np.array(history.epoch), np.log(np.array(history.history['loss'])))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
