import pandas as pd
import numpy as np
import tensorflow as tf


# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Y_train = train["label"]
X_train=train.drop(labels = ["label"],axis = 1)

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0


#expand 1 more dimention as 1 for colour channel gray
X_train = X_train.reshape(X_train.shape[0], 28, 28)


print(X_train.head())