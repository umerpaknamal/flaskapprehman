# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:43:16 2023

@author: hafsa
"""

# app.py

import numpy as np
import numpy
from flask import Flask, render_template, request
from PIL import Image
from keras.models import load_model

app = Flask(__name__)

# Load the trained model to classify herbs
model = load_model('C:/FYP FINAL/herb_main_model.h5')

# Dictionary to label all herb classes.
classes = {
    0: 'Scientific Name = Colonyth\nLocal Name = Bitter Apple\nSpray for Cotton = Oxyfluerfen Glyphosate\nSpray for Wheat= Paraquat',
    1: 'Scientific Name = Cyperous Roundus\nLocal Name = Della\nSpray for Cotton = S-Metolachor \nSpray for Wheat= glyphosate\nSpray for Rice= Ethoxy Sulfuron hosate',
    2: 'Scientific Name = Phalaris Minor\nLocal Name = Dumbi Setti\nSpray for Cotton = S-Metolachor \nSpray for Wheat=  Acetochlor \nSpray for Maize = Atrazine+ S metolachor ',
    3: 'Scientific Name = Convolvulus Arvenis\nLocal Name = Bathu\nSpray for Cotton = Pendimethal \nSpray for Wheat=  Bromoxynil \nSpray for Maize = Fluroxypyr+Mcpa ',
    4: 'Scientific Name = Famrica Indica\nLocal Name = Shahtara\nSpray for Cotton = B-romoxynil \nSpray for Wheat=  Thifensulfuron \nSpray for Maize = Fluroxypyr+Mcpa ',
    5: 'Scientific Name = Convolvulus\nLocal Name = lehli\nSpray for Cotton = B-romoxynil \nSpray for Wheat=  TCarfendrazone \nSpray for Maize = Fluroxypyr+Mcpa'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    image = Image.open(file)
    image = image.resize((250, 250))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image) / 255.0  # Normalize the image if required
    
    pred = model.predict(image)
    class_index = numpy.argmax(pred)
    sign = classes[class_index]
    
    return sign

if __name__ == '__main__':
    app.run(debug=True)
