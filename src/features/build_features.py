import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.base import BaseEstimator, TransformerMixin

# 2.1 read data
df_train = pd.read_csv("../../data/raw/train.csv")
X = df_train.drop("label", axis=1)
y = df_train["label"]

# 2.2 Create train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)


# 2.3 Preprocessing pipeline
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self._feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self._feature_names]
