import math
import numpy as np


def normalize_mean_std(x, mean, std):
    return (x - mean) / std


def denormalize_mean_std(x, mean, std):
    return x * std + mean


def normalize_mean(x, mean):
    return x / mean


def denormalize_mean(x, mean):
    return x * mean


def test_data(input_data, output):
    for i in range(input_data.shape[0]):
        if math.isnan(input_data[i, 0]):
            print('Input data contains nan value: ', i)
        if math.isnan(output[i]):
            print('Output data contains nan value: ', i)
