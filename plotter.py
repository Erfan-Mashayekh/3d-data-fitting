import matplotlib.pyplot as plt
import numpy as np

from utilities import *


def plot_loss(history):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.array(history.epoch), np.log(np.array(history.history['loss'])))
    ax.title('model loss')
    ax.ylabel('loss')
    ax.xlabel('epoch')
    fig.tight_layout()
    plt.savefig('loss.png', dpi=300)

