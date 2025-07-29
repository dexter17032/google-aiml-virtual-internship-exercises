import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#preparing data

cat_training_dir = r'D:\google_ai_ml\cat_or_dog_dataset\train\cats'
dog_training_dir = r'D:\google_ai_ml\cat_or_dog_dataset\train\dogs'
cat_testing_dir = r'D:\google_ai_ml\cat_or_dog_dataset\test\cats'
cat_testing_dir = r'D:\google_ai_ml\cat_or_dog_dataset\test\dogs'

train_data_gen = ImageDataGenerator(1./255)
test_data_gen = ImageDataGenerator(1./255)

train_generator = train_data_gen.flow_from_directory(
    r'D:\google_ai_ml\cat_or_dog_dataset\train',
    target_size=(200,200),
    batch_size=100,
    class_mode = 'binary'
)

test_generator = test_data_gen.flow_from_directory(
    r'D:\google_ai_ml\cat_or_dog_dataset\test',
    target_size=(200,200),
    batch_size=100,
    class_mode = 'binary'
)

layer = tf.keras.layers

model = keras.models.Sequential([
    layer.Conv2D(32 , (5,5) , activation='relu' , input_shape=(200,200,3)),
    layer.MaxPooling2D(2,2),
    
    layer.Conv2D(32 , (4,4) , activation = 'sigmoid'),
    layer.MaxPooling2D(2,2),
    
    layer.Conv2D(32 , (3,3) , activation = 'relu'),
    layer.MaxPooling2D(2,2),
    
    layer.Flatten(),
    layer.Dense(4096 , activation='sigmoid'),
    layer.Dense(1 , activation = 'relu'),
])

model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss = 'binary_crossentropy',
    
    metrics = ['accuracy'],
)


model.fit(
    train_generator,
    epochs = 10,
)

model.evaluate(test_generator)



