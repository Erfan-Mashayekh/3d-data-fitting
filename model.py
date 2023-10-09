import matplotlib.pyplot as plt
import os
from tensorflow import keras
from keras.models import model_from_json
from plotter import plot_loss
from utilities import *


def generate_model(control_dict):
    # Create the model
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_dim=2))
    model.add(keras.layers.Dense(units=3, activation=control_dict["activation"]))
    model.add(keras.layers.Dense(units=3, activation=control_dict["activation"]))
    model.add(keras.layers.Dense(units=4, activation=control_dict["activation"]))
    model.add(keras.layers.Dense(units=4, activation=control_dict["activation"]))
    model.add(keras.layers.Dense(units=3, activation=control_dict["activation"]))
    model.add(keras.layers.Dense(units=1, activation=control_dict["activation"]))
    model.compile(loss=control_dict["loss"], optimizer=control_dict["optimizer"], metrics=[control_dict["metrics"]])
    print(model.summary())
    return model


def train_model(model, inputs_norm, output_norm, control_dict):
    history = model.fit(inputs_norm, output_norm, epochs=int(control_dict["epochs"]), verbose=0)
    plot_loss(history)
    return history


def compute_plot_error(input_1, input_2, output, dataset, model, y_ref, control_dict):
    x_plot = np.zeros(input_1.shape)
    z_plot = np.zeros(output.shape)

    for i in range(input_1.size):
        if dataset[control_dict['input_2']][i] == y_ref:
            x_plot[i] = input_1[i]
            z_plot[i] = output[i]

    X = normalize_mean_std(x_plot, input_1.mean(), input_1.std())
    Y = (y_ref * np.ones(X.shape) - input_2.mean()) / input_2.std()
    input_data_plot = np.array([X, Y]).T
    Z = model.predict(input_data_plot)
    Z = Z.reshape(X.shape)
    X_plot = denormalize_mean_std(X, input_1.mean(), input_1.std())
    Z_plot = denormalize_mean(Z, output.mean())

    # Display the dataset
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x_plot, z_plot, label="Data", s=2)
    ax.scatter(X_plot, Z_plot, label="Model Output", color='red', s=2)
    ax.set_xlabel(control_dict["input_1"])
    ax.set_ylabel(control_dict["output"])
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.grid()
    fig.tight_layout()
    plt.savefig('./output/solution-check.png', dpi=300)

    # Display the error
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(X_plot, (Z_plot - z_plot) / (z_plot + 1e-6) * 100.0, s=2)
    ax1.set_ylim(-40, 40)
    ax1.set_xlabel(control_dict["input_1"])
    ax1.set_ylabel("Relative Error (%)")
    ax1.grid()
    plt.savefig('./output/relative-error.png', dpi=300, bbox_inches="tight")

    print("Checkout ./output/relative-error.png and ./output/solution-check.png files")


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("./output/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./output/model.h5")
    print("Saved model to disk")


def load_model(control_dict):
    print('Pretrained model is available ...')
    # load json and create model
    json_file = open('./output/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    assert os.path.isfile('./output/model.h5'), "There are no available trained weights. Make sure the file model.h5 " \
                                                "exists! "
    # load weights into new model
    model.load_weights("./output/model.h5")
    model.compile(loss=control_dict["loss"], optimizer=control_dict["optimizer"], metrics=[control_dict["metrics"]])
    print("Loaded model from disk")
    return model


def write_parameters(model, input_1, input_2, output, control_dict):
    print("Write parameters to ./output/parameters.dat.")
    file = open("./output/parameters.dat", "w")

    file.write(f'{control_dict["input_1"]}: mean: {input_1.mean()}, std: {input_1.std()} \n')
    file.write(f'{control_dict["input_2"]}: mean: {input_2.mean()}, std: {input_2.std()} \n')
    file.write(f'{control_dict["output"]}: mean: {output.mean()}, std: {output.std()} \n')

    for layer_i in range(len(model.layers)):
        if layer_i > 0:
            weights = model.layers[layer_i].get_weights()[0].flatten().T
            biases = model.layers[layer_i].get_weights()[1]

            file.write(f'\n w{layer_i}: \n\n')
            for w in weights:
                file.write(f'{w} \n')
            file.write(f'\n b{layer_i}: \n\n')
            for b in biases:
                file.write(f'{b} \n')
    file.close()
