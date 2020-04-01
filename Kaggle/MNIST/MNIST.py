import pandas as pd
import numpy as np
import tensorflow as tf


# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Xtrain = train.iloc[:,1:]
ytrain = train.iloc[:,0]
print(Xtrain)

# Normalize the data
Xtrain=Xtrain.values.reshape(-1,28,28,1)
Xtrain = Xtrain / 255.0

test = test.values.reshape(-1,28,28,1) / 255.0


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
model.fit(Xtrain, ytrain, epochs=15)
#test_loss = model.evaluate(test)


#Predict result
result = model.predict(test)
results = np.argmax(result,axis = 1)
submission = pd.DataFrame({ 'ImageId' : list(range(1,len(results)+1)),
             'Label': results})

submission.to_csv('Submission file',index=False,header = True)
