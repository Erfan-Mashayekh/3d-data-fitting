# 3d-data-fitting

## Introduction

3D-data-fitting is a data management tool designed for working with datasets, especially tailored for machine learning models. It operates within a three-dimensional space, where the variables x and y represent inputs, and the variable z corresponds to the associated output values. The primary objective of this tool is to create a continuous 3D surface that accurately represents the relationships within the 3D dataset by employing deep learning techniques. This process involves training a model to understand the complex interactions between the input parameters x and y, ultimately leading to the creation of a predictive surface that can be invaluable for making data-driven decisions and predictions.

## Installation and Run
Make sure that the following packages and libraries are installed:
- `Python 3`
- `Numpy`: to work with arrays
- `Matplotlib`: to plot the results
- `Tensorflow`: to generate the neural network model
- `Pandas`: to read the csv file
- `h5py`: to save/load the trained model
- any `JSON` file reader library

Download 3d-data-fitting python code from the github link or use command: `git clone git@github.com:Erfan-Mashayekh/3d-data-fitting.git`
Run the code with the command `python3 make.py`

## Code Overview

Here is a short description of the files and directories in the repository: 3d-data-fitting
- `main.py`: Coordinates and all functions.
- `manage_data.py`: Reads the settings and the dataset and manages it
- `model.py`: Defines the model and training strategy
- `plotter.py`: Plots the the dataset
- `utilities.py`: Contains necessary functions such dataset normalizers.

modified by user:
- `input.JSON`: This file controls the inputs and outputs and the methods
- `data`: The directory containing the dataset
- `output`: The directory that stores the model and the final parameters and figures

## Usage

It is highly recommended to review the `how-to.pdf` document to gain a comprehensive understanding of how to use the code most efficiently.

## Conclusion

This code snippet is a simplified representation of a data management and machine learning workflow. To make it functional in a real project, you would need to define the missing functions and ensure the necessary data and resources are available. Make sure to adapt and customize it according to your specific use case and project requirements.
