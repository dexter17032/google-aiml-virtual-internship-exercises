import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

human_path = os.path.join(r'D:\google_ai_ml\horse_or_human_dataset\humans')
horses_path = os.path.join(r'D:\google_ai_ml\horse_or_human_dataset\horses')
layer = tf.keras.layers

print('total training horse images:', len(os.listdir(horses_path)))
print('total training human images:', len(os.listdir(human_path)))

model = tf.keras.models.Sequential([
    layer.Conv2D(32 , (5,5) ,activation='relu', input_shape = (300,300,3)),
    layer.MaxPooling2D(2,2),
    
    layer.Conv2D(32 , (3,3),activation='relu'),
    layer.MaxPooling2D(2,2),
    
    layer.Conv2D(32 , (3,3),activation='relu'),
    layer.MaxPooling2D(2,2),
    
    layer.Conv2D(32 , (3,3),activation='relu'),
    layer.MaxPooling2D(2,2),
    
    layer.Flatten(),
    layer.Dense(1024,activation='relu'),
    layer.Dense(1,activation = 'sigmoid'), 
])

model.compile(loss='binary_crossentropy' , optimizer='adam',metrics=['accuracy'])

training_data_genertion = ImageDataGenerator(1./255)

train_generator = training_data_genertion.flow_from_directory(r'D:\google_ai_ml\horse_or_human_dataset',target_size=(300,300),batch_size=128,class_mode='binary')

model.fit(train_generator,steps_per_epoch=8,epochs=15)