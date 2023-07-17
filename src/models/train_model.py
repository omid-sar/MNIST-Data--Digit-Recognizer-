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
import keras_tuner as kt


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
    x = layers.Conv2D(32, (5, 5), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, (5, 5), activation="relu")(x)
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
            32 * (2**i), (5, 5), activation=conv_activation_functions[i]
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
    X_train,
    y_train,
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
model2 = build_model()
train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)

steps_per_epoch = train_generator.n // train_generator.batch_size
print(train_generator.n, train_generator.batch_size, steps_per_epoch)

model_checkpoint_path = "../../models/augmented_best_model.h5"

augmented_callbacks_list = [
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

history2 = model2.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=40,
    steps_per_epoch=steps_per_epoch,
    callbacks=augmented_callbacks_list,
    verbose=2,
)

# Create a new figure and a subplots grid
fig, axs = plt.subplots(3, 1)

# Plot loss
axs[0].plot(history2.history["loss"])
axs[0].plot(history2.history["val_loss"])
axs[0].set_title("Model loss")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].legend(["Loss", "Val loss"])

# Plot accuracy
axs[1].plot(history2.history["accuracy"])
axs[1].plot(history2.history["val_accuracy"])
axs[1].set_title("Model accuracy")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
axs[1].legend(["Accuracy", "Val Accuracy"])

# Plot F1 score
axs[2].plot(history2.history["f1_m"], linewidth=5)
axs[2].plot(history2.history["val_f1_m"])
axs[2].set_title("Model F1")
axs[2].set_xlabel("Epochs")
axs[2].set_ylabel("F1")
axs[2].legend(["Training f1", "Validation f1"])

# Display the figure
plt.tight_layout()
plt.show()


# ------------------------ 6. Hyperparameter Tuning ----------------------------

"""
After incorporating a data augmentation technique, our next step is to identify the ideal 
hyper-parameter configurations that can optimize the performance of our model.
To accomplish this, we will utilize the keras-tuner library, which offers
a range of tuning algorithms such as RandomSearch and HyperBand.
"""

# 6.1 Building hyper-parameter model


def build_model_hp(hp):
    inp = keras.layers.Input(shape=[28, 28, 1])

    dropout = hp.Choice("conv_block_dropout", [0.125, 0.25, 0.375, 0.5])
    conv_kernel_size = hp.Choice("conv_kernel_size", [3, 5])

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
        metrics=["accuracy", precision_m, recall_m, f1_m],
    )

    return model


# 6.2 RandomSearch HyperParametres
#
# for final hyperparametrs tunning, i wll set MAX_TRIALS= 100 or 150
#
#

tuner = kt.RandomSearch(
    hypermodel=build_model_hp,
    objective="val_loss",
    max_trials=100,
    overwrite=False,
    directory="../../models/random_search",
    project_name="random_search_trials",
)

tuner.search_space_summary()

"""
tuner.search(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=30,
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
"""

top_model_random = tuner.get_best_models(1)[0]
top_model_hps_random = tuner.get_best_hyperparameters(1)[0]
best_hyperparameters_random = tuner.get_best_hyperparameters()[0].values

print(top_model_hps_random.values)
top_model_random.summary()

# 6.2.2 save RandomSearch model and hyperparametrs
top_model_random.save("../../models/random_search/top_model_random.h5")

with open("../../models/random_search/hyperparameters_random.pkl", "wb") as f:
    pickle.dump(best_hyperparameters_random, f)


# 6.3 HyperBand HyperParameters
#
#
# for final hyperparametrs tunning, i wll set MAX_EPOCHS= 50 and executions_per_trial=2
#
#
tuner2 = kt.Hyperband(
    hypermodel=build_model_hp,
    objective="val_loss",
    max_epochs=50,
    executions_per_trial=2,
    overwrite=False,
    directory="../../models/hyper_band",
    project_name="hyperband_results_trials",
)

tuner2.search_space_summary()

"""
tuner2.search(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=30,
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
"""


top_model_hyperband = tuner2.get_best_models(1)[0]
top_model_hps_hyperband = tuner2.get_best_hyperparameters(1)[0]
best_hyperparameters_hyperband = tuner2.get_best_hyperparameters()[0].values

print(top_model_hps_hyperband.values)
top_model_hyperband.summary()

# 6.2.2 save HyperBand model and hyperparametrs
top_model_hyperband.save("../../models/hyper_band/top_model_hyperband.h5")

with open("../../models/hyper_band/hyperparameters_hyperband.pkl", "wb") as f:
    pickle.dump(best_hyperparameters_hyperband, f)

# ------------------------------ 7. Ensembling ---------------------------------
"""
Let's proceed to our final strategy, known as Ensembling.

Ensembling is a process where we use multiple models together to make predictions,
with the aim of getting better results. The idea is that using three models together 
should provide better results than using just one model alone. This is because each model 
will have its own strengths and weaknesses, and the weaknesses of one model can be 
compensated by the strengths of another model.

"""

# 7.1 Ensembling top models from hyper-parameter tuning
top_2_models = tuner2.get_best_models(2)
top_3_models = tuner2.get_best_models(3)
top_5_models = tuner2.get_best_models(5)
top_10_models = tuner2.get_best_models(10)

# We have now grouped top models into 4 groups; top 2, top 3, top 5 and top 10.
# We now need to find which group has highest validation accuracy.


def ensemble_models(models, data):
    results = np.zeros((data.shape[0], 10))
    for i in range(len(models)):
        results = results + models[i].predict(data)

    results = np.argmax(results, axis=1)
    results = pd.Series(results, name="Label")
    return results


# We've developed a function for a voting classifier to implement our final ensemble prediction.
# For every data point, each model casts a vote for a certain outcome. The outcome that garners
# the most votes ultimately becomes the final prediction for that specific data point.


from sklearn.metrics import accuracy_score

y_val_true = np.argmax(y_val, axis=1)


results = ensemble_models(top_2_models, X_val)
results = pd.concat(
    [pd.Series(np.arange(1, X_val.shape[0] + 1, 1), name="ImageId"), results], axis=1
)
print("Accuracy", accuracy_score(y_val_true, results["Label"].values))


results = ensemble_models(top_3_models, X_val)
results = pd.concat(
    [pd.Series(np.arange(1, X_val.shape[0] + 1, 1), name="ImageId"), results], axis=1
)
print("Accuracy", accuracy_score(y_val_true, results["Label"].values))


results = ensemble_models(top_5_models, X_val)
results = pd.concat(
    [pd.Series(np.arange(1, X_val.shape[0] + 1, 1), name="ImageId"), results], axis=1
)
print("Accuracy", accuracy_score(y_val_true, results["Label"].values))

results = ensemble_models(top_10_models, X_val)
results = pd.concat(
    [pd.Series(np.arange(1, X_val.shape[0] + 1, 1), name="ImageId"), results], axis=1
)
print("Accuracy", accuracy_score(y_val_true, results["Label"].values))


# After checking the accuracy score of the differents groups on the validation data, we can see
# top 2 and top 3 groups have the highest score. We will choose the top 2 ensembled model for
# computational efficiency.


# 7.2 Save the top 2 models
results = ensemble_models(top_2_models, X_test)
results = pd.concat(
    [pd.Series(np.arange(1, X_test.shape[0] + 1, 1), name="ImageId"), results], axis=1
)
results.to_csv("../../models/test_results_ensemble.csv", index=False)

# We will compute the accuracy score for the top 2 ensembled models for comparison later.


# 7.3 Ensembling on the best model from hyper-parameter tuning.

"""
Let's explore an alternative approach to ensemble modeling.

Rather than composing the top models from our tuning search, which feature varied configurations,
we aim to ensemble using only our highest-performing model. The strategy is to train this model 
multiple times. So, here's our revised concept:

We'll carry out multiple training rounds with our optimal model. Each iteration will involve 
a unique pairing of training and validation data, thereby ensuring comprehensive training across
the entire dataset.
"""

# The best network configuration from our hyperparameter tuning search is as follows:
{
    "conv_block_dropout": 0.25,
    "conv_kernel_size": 3,
    "n_conv_blocks": 4,
    "filter_combination_choice": 3,
    "n_fc_layers": 1,
    "fc_units_combination_choice": 0,
    "fc_dropout": 0.25,
    "tuner/epochs": 50,
    "tuner/initial_epoch": 17,
    "tuner/bracket": 2,
    "tuner/round": 2,
    "tuner/trial_id": "0067",
}


def build_best_model():
    inp = keras.layers.Input(shape=[28, 28, 1])

    conv_block_dropout = 0.25  # "conv_block_dropout": 0.25
    conv_kernel_size = 3  # "conv_kernel_size": 3
    n_conv_blocks = 4  # "n_conv_blocks": 4
    filter_combination_choice = 3  # "filter_combination_choice": 3

    # We select the appropriate filter settings
    filter_settings = [
        128,
        128,
        256,
        256,
    ]  # Based on filter_combination_choice and n_conv_blocks

    # Convolutional blocks
    for i in range(n_conv_blocks):
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
        x = keras.layers.Dropout(conv_block_dropout)(x)

    x = keras.layers.Flatten()(x)

    n_fc_layers = 1  # "n_fc_layers": 1
    fc_units_combination_choice = 0  # "fc_units_combination_choice": 0

    # We select the appropriate fully connected units settings
    fc_units = [128]  # Based on n_fc_layers and fc_units_combination_choice

    # Fully connected layers
    fc_dropout = 0.25  # "fc_dropout": 0.25
    for j in range(n_fc_layers):
        x = keras.layers.Dense(fc_units[j], activation="relu")(x)
        x = keras.layers.Dropout(fc_dropout)(x)

    out = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inp, outputs=out)

    # Compile the model
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy", precision_m, recall_m, f1_m],
    )

    return model


model = build_best_model()

# We built our best model. So we will create 10 copies of this model and train them.

n_models = 10
models = [0] * 10
for i in range(n_models):
    models[i] = build_best_model()

sys.path.append("../../features/")
from build_features import feature_pipeline, target_pipeline

df_train = pd.read_csv("../../data/raw/train.csv")
X = df_train.drop("label", axis=1)
y = df_train["label"]


historys = [0] * n_models
for i in range(n_models):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1
    )  # Random data each time

    X_train = feature_pipeline.fit_transform(X_train)
    y_train = target_pipeline.fit_transform(y_train.values.reshape(-1, 1))
    y_train = y_train.toarray()

    X_val = feature_pipeline.fit_transform(X_val)
    y_val = target_pipeline.fit_transform(y_val.values.reshape(-1, 1))
    y_val = y_val.toarray()

    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)

    historys[i] = models[i].fit(
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
        verbose=0,
    )

    models[i].save("../../models/top_model_ensemble/model_{}".format(str(i)))

    idx = np.argmin(historys[i].history["val_loss"])
    print(
        "Model: {} || Training loss: {}, Validation loss: {}, Training accuracy: {}, Validation accuracy: {}".format(
            i + 1,
            round(historys[i].history["loss"][idx], 4),
            round(historys[i].history["val_loss"][idx], 4),
            round(historys[i].history["accuracy"][idx], 4),
            round(historys[i].history["val_accuracy"][idx], 4),
        )
    )

# Ok, this took some time (~8 hour), but the models are now trained and ready for testing.

# Load models if needed
try:
    models
except:
    n_models = 10
    models = [0] * 10
    for i in range(n_models):
        models[i] = keras.models.load_model(
            "../../models/top_model_ensemble/model_{}".format(str(i)),
            custom_objects={
                "f1_m": f1_m,
                "precision_m": precision_m,
                "recall_m": recall_m,
            },
        )

# models_ordered
models_val_idxs = np.argsort(models_val_acc)[::-1]
models_ordered = np.array(models)[models_val_idxs]

# Now lets group the models into; best_model, top 2, top 3, top 5 and top 10.

models_ordered_1 = models_ordered[0:1]
models_ordered_2 = models_ordered[0:2]
models_ordered_3 = models_ordered[0:3]
models_ordered_5 = models_ordered[0:5]
models_ordered_10 = models_ordered[0:10]

# Lets check the scores on validation set for each group.

y_val_true = np.argmax(y_val, axis=1)

results = ensemble_models(models_ordered_1, X_val)
results = pd.concat(
    [pd.Series(np.arange(1, X_val.shape[0] + 1, 1), name="ImageId"), results], axis=1
)
print("Accuracy", accuracy_score(y_val_true, results["Label"].values))

results = ensemble_models(models_ordered_2, X_val)
results = pd.concat(
    [pd.Series(np.arange(1, X_val.shape[0] + 1, 1), name="ImageId"), results], axis=1
)
print("Accuracy", accuracy_score(y_val_true, results["Label"].values))

results = ensemble_models(models_ordered_3, X_val)
results = pd.concat(
    [pd.Series(np.arange(1, X_val.shape[0] + 1, 1), name="ImageId"), results], axis=1
)
print("Accuracy", accuracy_score(y_val_true, results["Label"].values))

results = ensemble_models(models_ordered_5, X_val)
results = pd.concat(
    [pd.Series(np.arange(1, X_val.shape[0] + 1, 1), name="ImageId"), results], axis=1
)
print("Accuracy", accuracy_score(y_val_true, results["Label"].values))

results = ensemble_models(models_ordered_10, X_val)
results = pd.concat(
    [pd.Series(np.arange(1, X_val.shape[0] + 1, 1), name="ImageId"), results], axis=1
)
print("Accuracy", accuracy_score(y_val_true, results["Label"].values))
