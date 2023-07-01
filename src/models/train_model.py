import numpy as np
import pandas as pd
import os
import sys
import pickle

from keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import Precision, Recall
import tensorflow.keras.layers as layers  # for building layers
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


import matplotlib.pyplot as plt

sys.path.append("../../src/visualization/")
from plot_settings import set_plot_style

set_plot_style()


# ---------------------------------- 1. read data-------------------------------

with open("../../data/processed/data.pkl", "rb") as file:
    X_train, y_train, X_val, y_val, X_test = pickle.load(file)
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape

# Use a subset of data to train the model
X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train, y_train, test_size=0.9, stratify=y_train, random_state=42
)

# Now, you can use X_train_sample and y_train_sample to train your model.

# ----------------------------- 1.2 read defined metrics from metrics.py -------------------------------
# Precision (using keras backend)
sys.path.append("../../models")
from metrics import precision_m, recall_m, f1_m

# ----------------------------- 2. Build Network(Ver1) -------------------------------
# 3.3.1 Build model based on Adam optimizer (with default learning rate) and sparse_categorical_crossentropy loss function


"""
def build_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


model = build_model()
build_model().summary()
print(build_model().summary())
plot_model(model)
"""


# ----------------------------- 3. Build Network(Ver2) -------------------------------
# 3.3.1 Build model based on Adam optimizer (with default learning rate) and sparse_categorical_crossentropy loss function
def build_model(
    input_shape=(28, 28, 1),
    dropout_rates=[0.25, 0.25, 0.5],
    learning_rate=0.001,
    optimizer=Adam,
    early_stopping_patience=10,
    model_checkpoint_path="model.h5",
    num_conv_layers=2,
    num_dense_layers=1,
    conv_activation_functions=["relu", "relu"],
    dense_activation_functions=["relu"],
    pooling_sizes=[(2, 2), (2, 2)],
):
    assert (
        num_conv_layers == len(conv_activation_functions) == len(pooling_sizes)
    ), "The length of activation functions and pooling sizes lists must equal the number of convolutional layers"
    assert num_dense_layers == len(
        dense_activation_functions
    ), "The length of activation functions list must equal the number of dense layers"

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    x = inputs
    # Define the convolutional layers
    for i in range(num_conv_layers):
        x = layers.Conv2D(
            32 * (2**i), (3, 3), activation=conv_activation_functions[i]
        )(x)
        x = layers.MaxPooling2D(pooling_sizes[i])(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rates[i])(x)

    x = layers.Flatten()(x)

    # Define the dense layers
    for i in range(num_dense_layers):
        x = layers.Dense(128 * (2**i), activation=dense_activation_functions[i])(x)
        x = layers.Dropout(dropout_rates[i + num_conv_layers])(x)

    # Define the output layer
    outputs = layers.Dense(10, activation="softmax")(x)
    # Define the model and compile it
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", f1_m, precision_m, recall_m],
    )

    return model


model = build_model()
build_model().summary()
print(build_model().summary())
plot_model(model)


# ----------------------------- 4. Train model ---------------------------------
# 4.1 fit model
# 4.1.1 callbacks

model_checkpoint_path = "../../models/best_model.h5"

callbacks_list = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.002,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=3, verbose=1, mode="min"
    ),
    ModelCheckpoint(
        model_checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
        mode="min",
        histogram_freq=1,
    ),
]
# 4.1.2 parameters
batch_size = 32
epochs = 40

history = model.fit(
    X_train_sample,
    y_train_sample,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=2,
)

# Create a new figure and a subplots grid
fig, axs = plt.subplots(3, 1)

# Plot loss
axs[0].plot(history.history["loss"])
axs[0].plot(history.history["val_loss"])
axs[0].set_title("Model loss")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].legend(["Loss", "Val loss"])

# Plot accuracy
axs[1].plot(history.history["accuracy"])
axs[1].plot(history.history["val_accuracy"])
axs[1].set_title("Model accuracy")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
axs[1].legend(["Accuracy", "Val Accuracy"])

# Plot F1 score
axs[2].plot(history.history["f1_m"], linewidth=5)
axs[2].plot(history.history["val_f1_m"])
axs[2].set_title("Model F1")
axs[2].set_xlabel("Epochs")
axs[2].set_ylabel("F1")
axs[2].legend(["Training f1", "Validation f1"])

# Display the figure
plt.tight_layout()
plt.show()


# 4.1.3 Save model
model.save("../../models/final_model.h5")
# 3.5.2 Save history
with open("../../models/history.pkl", "wb") as file:
    pickle.dump(history.history, file)


# ------------------------ 5. Training Augmented Data -------------------------
"""
Through data augmentation, we can create artificial data, or in this case, new images,
by slightly altering the images in our training set via various transformations.
In this particular notebook, the transformations we'll employ involve shifting,
rotating, and zooming the images to fabricate new instances.

Data augmentation provides a significant advantage as it acts as a regularizer, 
thus mitigating overfitting during model training. This is attributed to the increase 
in artificially created images which prevent the model from overfitting to 
particular examples and force it to generalize.
Consequently, the model becomes more resilient and 
generally exhibits improved overall performance.
"""


# 5.1.1 Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1
)

"""
ImageDataGenerator is an exceptional class in Keras that facilitates real-time 
image augmentation while our model is in the training phase. This capability means 
we can feed it into our model, and it will continuously generate new, 
augmented images in batches during the training process.
"""
model2, _, _ = build_model()
train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
steps_per_epoch = train_generator.n // train_generator.batch_size
print(train_generator.n, train_generator.batch_size)
steps_per_epoch
history2 = model2.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=40,
    steps_per_epoch=steps_per_epoch,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            min_delta=0.005,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3),
    ],
)


# ------------------------ 6. Hyperparameter Tuning ----------------------------
def build_model_hp(hp):
    inp = keras.layers.Input(shape=[28, 28, 1])

    dropout = hp.Choice("conv_block_dropout", [0.125, 0.25, 0.375, 0.5])
    conv_kernel_size = hp.Choice(
        "conv_kernel_size", [5]
    )  # Kernel size 5 is optimal after mutliple testing experiments

    n_layers = hp.Choice("n_conv_blocks", [2, 3, 4])

    filter_choice = hp.Choice("filter_combination_choice", [0, 1, 2, 3])

    filter_combinations_2 = [[16, 32], [32, 64], [64, 128], [128, 256]]
    filter_combinations_3 = [[16, 32, 48], [16, 32, 64], [32, 64, 128], [64, 128, 256]]
    filter_combinations_4 = [
        [16, 16, 32, 32],
        [32, 32, 64, 64],
        [64, 64, 128, 128],
        [128, 128, 256, 256],
    ]

    if n_layers == 2:
        filter_settings = filter_combinations_2[filter_choice]
    elif n_layers == 3:
        filter_settings = filter_combinations_3[filter_choice]
    elif n_layers == 4:
        filter_settings = filter_combinations_4[filter_choice]

    for i in range(n_layers):
        if i == 0:
            x = keras.layers.Conv2D(
                filters=filter_settings[i],
                kernel_size=conv_kernel_size,
                strides=1,
                padding="SAME",
                activation="relu",
            )(inp)
        else:
            x = keras.layers.Conv2D(
                filters=filter_settings[i],
                kernel_size=conv_kernel_size,
                strides=1,
                padding="SAME",
                activation="relu",
            )(x)

        x = keras.layers.MaxPool2D(pool_size=2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Flatten()(x)

    n_fc_layers = hp.Choice("n_fc_layers", [1, 2, 3])

    fc_choice = hp.Choice("fc_units_combination_choice", [0, 1])

    fc_combinations_1 = [[128], [256]]
    fc_combinations_2 = [[128, 64], [256, 128]]
    fc_combinations_3 = [[512, 256, 128], [256, 128, 64]]

    if n_fc_layers == 1:
        fc_units = fc_combinations_1[fc_choice]
    elif n_fc_layers == 2:
        fc_units = fc_combinations_2[fc_choice]
    elif n_fc_layers == 3:
        fc_units = fc_combinations_3[fc_choice]

    for j in range(n_fc_layers):
        x = keras.layers.Dense(fc_units[j], activation="relu")(x)
        x = keras.layers.Dropout(hp.Choice("fc_dropout", [0.125, 0.25, 0.5]))(x)

    out = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inp, outputs=out)

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy", f1_metric, recall_metric, precision_metric],
    )

    return model
