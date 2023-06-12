import numpy as np
import pandas as pd
import os
import sys
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import Precision, Recall
import tensorflow.keras.layers as layers  # for building layers
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ----------------------------------3.1 read data-------------------------------

with open("../../data/processed/data.pkl", "rb") as file:
    X_train, y_train, X_val, y_val, X_test = pickle.load(file)

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
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


model = build_model()
build_model().summary()
print(build_model().summary())
plot_model(model)


# ----------------------------- 3.2 Build Network(Ver2) -------------------------------
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
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Define early stopping and model checkpointing
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min",
        restore_best_weights=True,
    )
    model_checkpoint = ModelCheckpoint(
        model_checkpoint_path, monitor="val_loss", mode="min", save_best_only=True
    )

    return model, [early_stopping, model_checkpoint]


model = built_model()[0]
built_model()[0].summary()
print(built_model()[0].summary())
plot_model(model)


# ----------------------------- 3.3 Train model ---------------------------------
# 3.3.1 Train model
history = model.fit()


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=batch_size,
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
