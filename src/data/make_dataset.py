import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


sample = pd.read_csv("../../data/raw/sample_submission.csv")
train = pd.read_csv("../../data/raw/train.csv")
test = pd.read_csv("../../data/raw/test.csv")

test.info()
train.info()
df.info()
