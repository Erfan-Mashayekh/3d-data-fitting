import math
import numpy as np

def normalize_mean_std(x):
    return (x - x.mean()) / x.std()

def normalize_mean(x):
    return x / x.mean()

def test_data(input_data, output):
    for i in range(input_data.shape[0]):
        if (math.isnan(input_data[i, 0])):
            print('Input data contains nan value: ', i)
        if (math.isnan(output[i])):
            print('Output data contains nan value: ', i)