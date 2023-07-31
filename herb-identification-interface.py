# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:56:48 2023

@author: hafsa
"""

import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy
# Load the trained model to classify herbs
from keras.models import load_model

model = load_model('C:/FYP FINAL/herb_main_model.h5')

# Dictionary to label all herb classes.
classes = {
    0: 'Scientific Name = Colonyth\nLocal Name = Bitter Apple\nSpray for Cotton = Oxyfluerfen Glyphosate\nSpray for Wheat= Paraquat' ,
    1: 'Scientific Name = Cyperous Roundus\nLocal Name = Della\nSpray for Cotton = S-Metolachor \nSpray for Wheat= glyphosate\nSpray for Rice= Ethoxy Sulfuron hosate',
    2: 'Scientific Name = Phalaris Minor\nLocal Name = Dumbi Setti\nSpray for Cotton = S-Metolachor \nSpray for Wheat=  Acetochlor \nSpray for Maize = Atrazine+ S metolachor ',
    3: 'Scientific Name = Convolvulus Arvenis\nLocal Name = Bathu\nSpray for Cotton = Pendimethal \nSpray for Wheat=  Bromoxynil \nSpray for Maize = Fluroxypyr+Mcpa ',
    4: 'Scientific Name = Famrica Indica\nLocal Name = Shahtara\nSpray for Cotton = B-romoxynil \nSpray for Wheat=  Thifensulfuron \nSpray for Maize = Fluroxypyr+Mcpa ',
    5: 'Scientific Name = Convolvulus\nLocal Name = lehli\nSpray for Cotton = B-romoxynil \nSpray for Wheat=  TCarfendrazone \nSpray for Maize = Fluroxypyr+Mcpa'
}

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Herbs Identification')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((250, 250))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image) / 255.0  # Normalize the image if required
    pred = model.predict(image)
    class_index = numpy.argmax(pred)
    sign = classes[class_index]
    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Unveiling the Botanical Treasure: Know Your Crop's Herbs", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
