import matplotlib.pyplot as plt

from tensorflow import keras
from plotter import plot_loss
from utilities import *


def generate_model():
    # Create the model
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_dim=2))
    model.add(keras.layers.Dense(units=3, activation='elu'))
    model.add(keras.layers.Dense(units=3, activation='elu'))
    model.add(keras.layers.Dense(units=3, activation='elu'))
    model.add(keras.layers.Dense(units=3, activation='elu'))
    model.add(keras.layers.Dense(units=1, activation='elu'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    print(model.summary())
    return model


def train_model(model, inputs, output_norm, control_dict):
    history = model.fit(inputs, output_norm, epochs=int(control_dict["epochs"]), verbose=0)
    plot_loss(history)
    return history


def compute_plot_error(input_1, input_2, output, dataset, model, y_ref, control_dict):
    x_plot = np.zeros(input_1.shape)
    z_plot = np.zeros(output.shape)

    for i in range(input_1.size):
        if dataset['p'][i] == y_ref:
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
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(control_dict["output"])
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.grid()
    fig.tight_layout()
    plt.savefig('solution-check.png', dpi=300)

    # Display the error
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.scatter(X_plot, (Z_plot - z_plot) / (z_plot + 1 - 6) * 100.0, s=2)
    ax1.set_ylim(-40, 40)
    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("Relative Error (%)")
    ax1.grid()
    plt.savefig('relative-error.png', dpi=300, bbox_inches="tight")


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
