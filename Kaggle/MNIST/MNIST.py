import pandas as pd
import numpy as np
import tensorflow as tf


# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Y_train = train["label"]
X_train=train.drop(labels = ["label"],axis = 1)

# Normalize the data
X_train=X_train.reshape(60000, 28, 28, 1)
X_train = X_train / 255.0
Y_train = Y_train.reshape(10000, 28, 28, 1)
Y_train=Y_train / 255.0

test = test / 255.0

#expand 1 more dimention as 1 for colour channel gray
X_train = X_train.reshape(X_train.shape[0], 28, 28)
print(X_train.head())


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, epochs=5)
#test_loss = model.evaluate(test_images, test_labels)