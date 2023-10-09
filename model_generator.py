from read_data import read_properties, read_input_data
from utilities import *
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

dataset = read_properties()
control_dict = read_input_data()

input_1 = dataset[control_dict["input_1"]] * float(control_dict["scale_input_1"])
input_2 = dataset[control_dict["input_2"]] * float(control_dict["scale_input_2"])
output = dataset[control_dict["output"]] * float(control_dict["scale_output"])

input_1_norm = normalize_mean_std(input_1)
input_2_norm = normalize_mean_std(input_2)
inputs = np.array([input_1_norm, input_2_norm]).T
output_norm = normalize_mean(output)
test_data(inputs, output_norm)

# Create the model
model = keras.Sequential()
model.add(keras.layers.Flatten(input_dim=2))
model.add(keras.layers.Dense(units = 3, activation = 'elu'))
model.add(keras.layers.Dense(units = 3, activation = 'elu'))
model.add(keras.layers.Dense(units = 3, activation = 'elu'))
model.add(keras.layers.Dense(units = 3, activation = 'elu'))
model.add(keras.layers.Dense(units = 1, activation = 'elu'))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
print(model.summary())

history = model.fit(inputs, output_norm, epochs=1000, verbose=0)
print(history.history.keys())
print(np.shape(history.history['loss']))

plt.plot(np.array(history.epoch), np.log(np.array(history.history['loss'])))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()