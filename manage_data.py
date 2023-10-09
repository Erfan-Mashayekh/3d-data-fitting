import pandas as pd
import json
from utilities import *


def read_properties():
    dataset = pd.read_csv('./data/CO2-data.csv')
    print("Dataset Description: \n", dataset.describe())
    return dataset


def read_input_data():
    f = open('input.json')
    data = json.load(f)
    f.close()
    return data


def generate_data(dataset, control_dict):
    input_1 = dataset[control_dict["input_1"]] * float(control_dict["scale_input_1"])
    input_2 = dataset[control_dict["input_2"]] * float(control_dict["scale_input_2"])
    output = dataset[control_dict["output"]] * float(control_dict["scale_output"])
    return input_1, input_2, output


def manage_data(input_1, input_2, output):
    input_1_norm = normalize_mean_std(input_1, input_1.mean(), input_1.std())
    input_2_norm = normalize_mean_std(input_2, input_2.mean(), input_2.std())
    inputs = np.array([input_1_norm, input_2_norm]).T
    output_norm = normalize_mean(output, output.mean())
    return inputs, output_norm
