import pandas as pd


def read_properties():
    dataset = pd.read_csv('./data/CO2-data.csv')
    print("Dataset Description: \n", dataset.describe())
    return dataset

