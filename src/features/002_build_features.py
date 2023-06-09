import pandas as pd
import numpy as np
import os
import sys
import pickle


# 2.1 read data
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("../../data/raw/train.csv")
X = df_train.drop("label", axis=1)
y = df_train["label"]

df_test = pd.read_csv("../../data/raw/test.csv")


# 2.2 Create train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)


# 2.3 Preprocessing pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


# Create a custom transformer to add channel dimension to the data. Since the data is black and white, we have only one channel.
class AddChannelDim(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.reshape(-1, 28, 28, 1)
        return X


# Create feature pipeline for preprocessing whcih includes adding channel dimension and scaling the data between 0 and 1
feature_pipeline = Pipeline(
    [("Normalize", MinMaxScaler()), ("Reshape", AddChannelDim())]
)
# create target pipeline for preprocessing which includes one hot encoding
target_pipeline = Pipeline([("OneHotEncode", OneHotEncoder())])

X_train = feature_pipeline.fit_transform(X_train)
y_train = target_pipeline.fit_transform(y_train.values.reshape(-1, 1))
print(y_train.shape, X_train.shape)
# y_train = target_pipeline.fit_transform(y_train.values.reshape(-1, 1)).toarray()

X_val = feature_pipeline.fit_transform(X_val)
y_val = target_pipeline.fit_transform(y_val.values.reshape(-1, 1)).toarray()
print(y_val.shape, X_val.shape)

X_test = feature_pipeline.fit_transform(df_test)
print(X_test.shape)

# 2.4 Save the preprocessed data
with open("../../data/processed/data.pkl", "wb") as file:
    pickle.dump((X_train, y_train, X_val, y_val, X_test), file)
