import matplotlib.pyplot as plt
import numpy as np

from utilities import *


def plot_loss(history):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.array(history.epoch), np.log(np.array(history.history['loss'])))
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    fig.tight_layout()
    plt.savefig('./output/loss.png', dpi=300)

