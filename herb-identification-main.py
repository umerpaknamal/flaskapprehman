# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 18:07:14 2023

@author: hafsa
"""

# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# Set the path to your dataset directory containing four subdirectories for each class.
data_dir = "G:/data_collection/pre-processed/AUGMENTATION/train"
classes = os.listdir(data_dir)

# Image dimensions and other hyperparameters
img_width, img_height = 250, 250
batch_size = 32
epochs = 20

# Creating the CNN model
model = Sequential()
# ... Your model architecture ...
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation to increase dataset size and generalize better
train_data_gen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
train_generator = train_data_gen.flow_from_directory(data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical', subset='training')
validation_generator = train_data_gen.flow_from_directory(data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical', subset='validation')

# Collect history during training for plotting
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Save the model for future use
model.save('herb_project_classification_model.h5')

# Plotting training accuracy and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])  # Now 'val_accuracy' will be available in history object
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting training loss and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])  # Now 'val_loss' will be available in history object
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
