import tensorflow as tf
from tensorflow import keras 
import numpy as np

(training_images, training_labels), (testing_images, testing_labels) = keras.datasets.mnist.load_data()
training_images = training_images/255.0
testing_images=testing_images/255.0

class myCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs ,logs = {}):
        if logs.get('accuracy')>0.94:
            print("97% accuracy reached ending training")
            self.model.stop_training = True

callbacks = myCallbacks()

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(128,activation=tf.nn.relu),tf.keras.layers.Dense(10,activation=tf.nn.softmax)])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images,training_labels,epochs=5)

print("testing...")

model.evaluate(testing_images,testing_labels)

classifications = model.predict(testing_images)


print(testing_labels[0])
print(classifications[0])

#128 neurons 3 epochs accuracy = 97.03% 2s
#512 neurons 3 epochs accuracy = 97.37% 3s
#1024 neurons 3 epochs accuracy = 97.51% 5s

