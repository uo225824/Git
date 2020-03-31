import tensorflow as tf
print(tf.__version__)

#The Fashion MNIST data is available directly in the tf.keras datasets API
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

#Set the values betwenn  0 and 1
training_images  = training_images / 255.0
test_images = test_images / 255.0

import tensorflow as tf
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.3):#'accuracy'
      print("\nReached 96% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()


#Model nn

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#Flatten just takes that square and turns it into a 1 dimensional set.
# Adds a layer of neurons
#Relu "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50,callbacks=[callbacks])


model.evaluate(test_images, test_labels)

from ann_visualizer.visualize import ann_viz;

ann_viz(model, title="My first neural network")