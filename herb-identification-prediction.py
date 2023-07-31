import os
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array  # Updated import statement
from PIL import Image

# Load the trained model
model = load_model('C:/FYP FINAL/herb_project_classification_model.h5')

# Define the class names (replace these with your actual class names)
class_names = ["0", "1", "2", "3","4", "5"]

# Function to preprocess an image for prediction
def preprocess_image(image_path, target_size=(250, 250)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Rescale pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions on a single image
def predict_single_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    return predicted_class, confidence

# Test the model on a single image
test_image_path = "G:/data_collection/pre-processed/AUGMENTATION/train/5/5 (83).jpeg"  # Replace with the path to your test image
predicted_class, confidence = predict_single_image(test_image_path)
print("Predicted Class:", predicted_class)
print("Confidence:", confidence)
model.save("herb_main_model.h5")