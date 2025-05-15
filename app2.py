from flask import Flask, render_template, request, url_for, Markup, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
from flask import *
import os
from werkzeug.utils import secure_filename
import label_image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend as K
import cv2

PEOPLE_FOLDER = os.path.join('static', 'people_photo')

app = Flask(__name__)
K.set_learning_phase(0)
# Disable eager execution to create a graph
tf.compat.v1.disable_eager_execution()

model = load_model('templates/mymodel.h5')
graph = tf.compat.v1.get_default_graph()

# Define a function to preprocess the input data

IMG_SIZE = 512

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    
def circle_crop(img, sigmaX=30):
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape
    min_dim = min(height, width)
    
    x = width // 2
    y = height // 2
    r = min_dim // 2
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), r, 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize the image to (512, 512)
    return img

def preprocess_image(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = circle_crop(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def load_image(image):
    print("load image")
    text = label_image.main(image)
    return text

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/index')
def index():
    return render_template('index.html')

img_width, img_height = 512, 512

@app.route('/predict', methods=['GET', 'POST'])
def upload():
   
   if request.method == 'POST':
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        print("file uploaded: ",file_path)
        f.save('./static/' + file_path)
        path='/static/' + file_path
        print("file uploaded: ",path)
        image = cv2.imread(file_path)
        
        global graph
        # Preprocess the image
        with graph.as_default():
            model = tf.keras.models.load_model('templates/mymodel.h5')
            img= preprocess_image(image)
            prediction = model.predict(img)
            print(prediction)
            predicted_class = np.argmax(prediction)

            return jsonify(predicted_class=str(predicted_class))

if __name__ == '__main__':
    app.run(debug=True)
