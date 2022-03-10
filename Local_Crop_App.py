# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 19:06:57 2022
Flask App for local Deployment
@author: Nimish
"""
from flask import Flask, render_template, session, redirect, url_for, jsonify, make_response
from flask_wtf import FlaskForm
#import pickle
#import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
#from wtforms import (StringField, BooleanField, DateTimeField, RadioField, SelectField, TextField, TextAreaField, SubmitField)
from wtforms import SubmitField
#from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileRequired, FileAllowed 
from flask_uploads import UploadSet, IMAGES, configure_uploads, patch_request_class
import cv2
from PIL import Image
import os
# from werkzeug import secure_filename

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'uploads') # you'll need to create a folder named uploads


photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app) # set maximum file size, default 16MB

# Create a WTForm Class
# We will get image input from the user
class ImageForm(FlaskForm):
    photo = FileField('Crop Image', validators = [FileRequired('File was empty!'), FileAllowed(photos, 'Images Only!')])
    submit = SubmitField('Submit')
    
def crop_pred(model, image):
    
    classes = ["Jute", "Maize", "Rice", "Sugarcane", "Wheat"]
    
    image = cv2.resize(image, (64,64)) #Input shape for the model
    
    image = image.astype("float") / 255.0 # Scale the pixel values from range 0-255 to range 0-1
    
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    preds = model.predict(image)[0] # Get prediction from the model
    
    result = dict()
    
    result[classes[0]] = round(float(list(preds)[0]), 3)
    result[classes[1]] = round(float(list(preds)[1]), 3)
    result[classes[2]] = round(float(list(preds)[2]), 3)
    result[classes[3]] = round(float(list(preds)[3]), 3)
    result[classes[4]] = round(float(list(preds)[4]), 3)
    
    return result

# Load the model
model_cnn = load_model("Crop_CNN.h5")
    
@app.route('/', methods = ['GET', 'POST'])
def index():
    
    form = ImageForm()  # Create an instance of form
    
    if form.validate_on_submit():  # If form is valid on submitting
        # Grab the image 
        
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
        filepath = photos.path(filename)
        session['ImgUrl'] = file_url
        session['ImgPath'] = filepath
        
        return redirect(url_for("predict_crop"))

    return render_template('home.html', form = form)

@app.route('/prediction')
def predict_crop():
    
    # Get the image
    file_url = session["ImgUrl"]
    imgpath = session['ImgPath']
    img = Image.open(imgpath)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = crop_pred(model = model_cnn, image = img)
    
    return render_template('thankyou.html', results = results, file_url = file_url)



if __name__ == '__main__':
    app.run('127.0.0.1', port = 5000, debug = True)

    
    
    
        

    
    
    
    

