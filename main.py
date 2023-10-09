import os
from manage_data import *
from model import *

dataset = read_properties()
control_dict = read_input_data()
input_1, input_2, output = generate_data(dataset, control_dict)
inputs_norm, output_norm = manage_data(input_1, input_2, output)
test_data(inputs_norm, output_norm)

if os.path.isfile('./model.json'):
    model = load_model()
else:
    print("Generate the model for the first time.")
    model = generate_model()

if int(control_dict["train_mod"]):
    print("Start training ...")
    train_model(model, inputs_norm, output_norm, control_dict)
    print("Finish training.")
    save_model(model)

y_ref = float(control_dict["plot_at_input_2"])
compute_plot_error(input_1, input_2, output, dataset, model, y_ref, control_dict)
write_parameters(model)
