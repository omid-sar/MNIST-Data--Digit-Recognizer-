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

# -------------------------- 3.2 Build and Train Network ------------


# 3.2.1 Build model based on Adam optimizer (with default learning rate) and sparse_categorical_crossentropy loss function
def built_model():
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


model = built_model()
print(built_model().summary())
plot_model(model)
