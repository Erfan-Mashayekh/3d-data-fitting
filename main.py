from read_data import read_properties, read_input_data
from model import *

dataset = read_properties()
control_dict = read_input_data()

input_1 = dataset[control_dict["input_1"]] * float(control_dict["scale_input_1"])
input_2 = dataset[control_dict["input_2"]] * float(control_dict["scale_input_2"])
output = dataset[control_dict["output"]] * float(control_dict["scale_output"])

input_1_norm = normalize_mean_std(input_1, input_1.mean(), input_1.std())
input_2_norm = normalize_mean_std(input_2, input_2.mean(), input_2.std())
inputs = np.array([input_1_norm, input_2_norm]).T
output_norm = normalize_mean(output, output.mean())
test_data(inputs, output_norm)

import os
from tensorflow.keras.models import model_from_json

if os.path.isfile('./model.json'):
    print('A pretrained model is available.')
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    assert os.path.isfile('./model.h5'), "There are no available trained weights. Make sure the file model.h5 exists!"
    # load weights into new model
    model.load_weights("model.h5")
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    print("Loaded model from disk")
else:
    print("Generate the model for the first time.")
    model = generate_model()

train_model(model, inputs, output_norm, control_dict)
y_ref = 0.2
compute_plot_error(input_1, input_2, output, dataset, model, y_ref, control_dict)
save_model(model)
