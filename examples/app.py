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

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('example_form.html')

@app.route('/process_form', methods=['POST'])
def process_form():
    name = request.form['name']
    return f'Hello, {name}!'