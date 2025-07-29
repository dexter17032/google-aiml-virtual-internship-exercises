import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

# Load and preprocess the Fashion MNIST data
(training_images, training_label), (testing_images, testing_label) = keras.datasets.fashion_mnist.load_data()

testing_images = testing_images.reshape(10000, 28, 28, 1)
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
testing_images = testing_images / 255.0

# Define the model
input_layer = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)

x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output_layer = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_label, epochs=1)

# Set up visualization of activations
f, axarr = plt.subplots(3, 4)
FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28
CONVOLUTION_NUMBER = 6  # Ensure this is within the range of your filter count

# Extract convolutional and pooling layer outputs for visualization
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.MaxPooling2D))]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

# Visualize the activations for selected images
for x in range(0, 4):
    # Display activations for FIRST_IMAGE
    f1 = activation_model.predict(testing_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0, x].grid(False)
    
    # Display activations for SECOND_IMAGE
    f2 = activation_model.predict(testing_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1, x].grid(False)
    
    # Display activations for THIRD_IMAGE
    f3 = activation_model.predict(testing_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2, x].grid(False)

plt.show()
#i dont know how tf this code works like the display part but else i get it