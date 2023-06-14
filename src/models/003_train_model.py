import numpy as np
import pandas as pd
import os
import sys
import pickle


from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import Precision, Recall
import tensorflow.keras.layers as layers  # for building layers
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


import matplotlib.pyplot as plt

sys.path.append("../../src/visualization/")
from plot_settings import set_plot_style

set_plot_style()

# ----------------------------------3.1 read data-------------------------------

with open("../../data/processed/data.pkl", "rb") as file:
    X_train, y_train, X_val, y_val, X_test = pickle.load(file)
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape

# Use a subset of data to train the model
X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train, y_train, test_size=0.9, stratify=y_train, random_state=42
)

# Now, you can use X_train_sample and y_train_sample to train your model.

"""
# Precision (using keras backend)
def precision_metric(y_true, y_pred):
    threshold = 0.5  # Training threshold 0.5
    y_pred_y = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())

    true_positives = K.sum(K.clip(y_true * y_pred, 0, 1))
    false_negatives = K.sum(K.clip(y_true * (1-y_pred), 0, 1))
    false_positives = K.sum(K.clip((1-y_true) * y_pred, 0, 1))
    true_negatives = K.sum(K.clip((1 - y_true) * (1-y_pred), 0, 1))

    precision = true_positives / (true_positives + false_positives + K.epsilon())
    return precision

# Recall (using keras backend)
def recall_metric(y_true, y_pred):
    threshold = 0.5 #Training threshold 0.5
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())

    true_positives = K.sum(K.clip(y_true * y_pred, 0, 1))
    false_negatives = K.sum(K.clip(y_true * (1-y_pred), 0, 1))
    false_positives = K.sum(K.clip((1-y_true) * y_pred, 0, 1))
    true_negatives = K.sum(K.clip((1 - y_true) * (1-y_pred), 0, 1))

    recall = true_positives / (true_positives + false_negatives + K.epsilon())
    return recall

# F1-score (using keras backend)
def f1_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (recall+precision+K.epsilon()))
    return f1

"""

"""
# instead of using keras backend, we can use tensorflow metrics
from tensorflow.keras.metrics import Precision, Recall
threshold = 0.5
precision = Precision(thresholds=threshold)
Recall = Recall(thresholds=threshold)

precision.update_state(y_train, y_pred)
Recall.update_state(y_train, y_pred)
print("Precision: ", precision.result().numpy())
print("Recall: ", Recall.result().numpy())
"""

# ----------------------------- 3.2 Build Network(Ver1) -------------------------------
# 3.2.1 Build model based on Adam optimizer (with default learning rate) and sparse_categorical_crossentropy loss function


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


# ----------------------------- 3.2 Build Network(Ver2) -------------------------------
# 3.2.1 Build model based on Adam optimizer (with default learning rate) and sparse_categorical_crossentropy loss function
def built_model(
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
        metrics=["accuracy"],
    )

    return model


model = built_model()
built_model().summary()
print(built_model().summary())
plot_model(model)


# ----------------------------- 3.3 Train model ---------------------------------
# 3.3.1 fit model
# 3.3.1.1 callbacks

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
# 3.3.1.2 parameters
batch_size = 32
epochs = 40

history = model.fit(
    X_train_sample,
    y_train_sample,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)


set_plot_style()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Val loss"])


plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.plot(history.history["f1_metric"], linewidth=5)
plt.plot(history.history["val_f1_metric"])
plt.title("Model F1")
plt.xlabel("Epochs")
plt.ylabel("F1")
plt.legend(["Training f1", "Validation f1"])
