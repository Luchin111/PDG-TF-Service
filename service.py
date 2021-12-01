# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com
#Reference: https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037
import numpy as np
import pandas as pd
import cv2 as cv 
from skimage import io
from PIL import Image 
import matplotlib.pylab as plt


#Import Flask
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS, cross_origin

#Import Keras
from keras.preprocessing import image

#Import python files
import numpy as np

import requests
import json
import os
from werkzeug.utils import secure_filename
from model_loader import cargarModelo

UPLOAD_FOLDER = '../images/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

port = int(os.getenv('PORT', 5000))
print ("Port recognized: ", port)

#Initialize the application service
app = Flask(__name__)
CORS(app)
global loaded_model, graph
loaded_model, graph = cargarModelo()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Define a route
@app.route('/')
@cross_origin()
def main_page():
	return '¡Servicio REST activo!'

@app.route('/model/compare')
@cross_origin()
def main_page():
	# Create a list to store the urls of the images
    urls = ["https://iiif.lib.ncsu.edu/iiif/0052574/full/800,/0/default.jpg",
        "https://iiif.lib.ncsu.edu/iiif/0016007/full/800,/0/default.jpg",
        "https://placekitten.com/800/571"]  
    # Read and display the image
    # loop over the image URLs, you could store several image urls in the list

    for url in urls:
        image = io.imread(url) 
        image_2 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    #--- take the absolute difference of the images ---
    res = cv2.absdiff(image, image_2)

    #--- convert the result to integer type ---
    res = res.astype(np.uint8)

    #--- find percentage difference based on number of pixels that are not zero ---
    percentage = (np.count_nonzero(res) * 100)/ res.size

    return percentage


@app.route('/model/covid19/', methods=['GET','POST'])
@cross_origin()
def default():
    data = {"success": False}
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
        file = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            print('No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #loading image
            filename = UPLOAD_FOLDER + '/' + filename
            print("\nfilename:",filename)

            image_to_predict = image.load_img(filename, target_size=(224, 224))
            test_image = image.img_to_array(image_to_predict)
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image.astype('float32')
            test_image /= 255

            with graph.as_default():
            	result = loaded_model.predict(test_image)[0][0]
            	# print(result)
            	
		# Resultados
            	prediction = 1 if (result >= 0.5) else 0
            	CLASSES = ['Normal', 'Con Cancer']

            	ClassPred = CLASSES[prediction]
            	ClassProb = result
            	
            	print("Pedicción:", ClassPred)
            	print("Prob: {:.2%}".format(ClassProb))

            	#Results as Json
            	data["predictions"] = []
            	r = {"label": ClassPred, "score": float(ClassProb)}
            	data["predictions"].append(r)

            	#Success
            	data["success"] = True

    return jsonify(data)

# Run de application
app.run(host='0.0.0.0',port=port, threaded=False)