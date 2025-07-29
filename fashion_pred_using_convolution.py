import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

(training_images, training_label), (testing_images,testing_label) = keras.datasets.fashion_mnist.load_data()


testing_images=testing_images.reshape(10000,28,28,1)
training_images = training_images.reshape(60000,28,28,1)
training_images = training_images/255.0
testing_images = testing_images/255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32 , (3,3) , activation='relu' , input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(32 , (3,3) , activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(10 , activation = 'softmax')
    ]
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images,training_label,epochs = 5)
model.evaluate(testing_images,testing_label)