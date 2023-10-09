import pandas as pd
import json

def read_properties():
    dataset = pd.read_csv('./data/CO2-data.csv')
    print("Dataset Description: \n", dataset.describe())
    return dataset

def read_input_data():
    f = open('input.json')
    data = json.load(f)
    f.close()
    return data