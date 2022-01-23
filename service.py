# Developed by Luis H. Medina M. luisinmedina@gmail.com
#Reference: https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037

import pandas as pd
import cv2  as cv
import imutiimls 
from skimage.metrics import structural_similarity as ssim
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

@app.route('/model/compare/', methods=['GET','POST'])
@cross_origin()
def compare():
    if request.method == "POST":
	# Create a list to store the urls of the images
                # check if the post request has the file part
        if 'fileA' not in request.files:
            print('No file part')
        if 'fileB' not in request.files:
            print('No file2 part')
        fileA = request.files['fileA']
        fileB = request.files['fileB']
        
        #loading images
        filename = secure_filename(fileA.filename)
        fileA.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        filename = UPLOAD_FOLDER + '/' + filename
        print("\nfilename:",filename)

        filename2 = secure_filename(fileB.filename)
        fileB.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

        filename2 = UPLOAD_FOLDER + '/' + filename2
        print("\nfilename2:",filename2)

       
        
        # if user does not select file, browser also submit a empty part without filename
        if fileA.filename == '' or fileB.filename == '':
            print('No selected files')

       
        print("fileA ",fileA.filename)
        print("fileB ",fileB.filename)

        original = cv.imread("1.png")
        new = cv.imread("2.png")
        original = imutils.resize(filename, height = 600)
        new = imutils.resize(filename2, height = 600)

        diff = original.copy()
        res = cv.absdiff(original, new, diff)
        print("\dif: ",res)
        #imagen = image.load_img(filename, target_size=(224, 224))
        #test_imageA = image.img_to_array(imagen)
        #test_imageA = np.expand_dims(test_imageA, axis = 0)
        #test_imageA = test_imageA.astype('float32')
        #test_imageA /= 255
        #imagen2 = image.load_img(filename2, target_size=(224, 224))
        #test_imageB = image.img_to_array(imagen2)
        #test_imageB = np.expand_dims(test_imageB, axis = 0)
        #test_imageB = test_imageB.astype('float32')
        #test_imageB /= 255
        #s = ssim(test_imageA, test_imageB)
        #print("\dif: ",s)

        #--- take the absolute difference of the images ---
        #res = cv.absdiff(imagen, imagen2)
        #print("\dif:",res)

        #--- convert the result to integer type ---
        #res = res.astype(np.uint8)

        #res2 = res2.astype(np.uint8)

        #--- find percentage difference based on number of pixels that are not zero ---
        #percentage = (np.count_nonzero(res) * 100)/ res.size
        #percentage2 = (np.count_nonzero(res2) * 100)/ res.size
        #print("\porcentaje:",percentage)
        #print("\porcentaje2:",percentage2)

    return 'ok '


@app.route('/model/cancer/', methods=['GET','POST'])
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
            	
            	print("Predicción:", ClassPred)
            	print("Prob: {:.2%}".format(ClassProb))

            	#Results as Json
            	data["predictions"] = []
            	r = {"label": ClassPred, "score": float(ClassProb)}
            	data["predictions"].append(r)

            	#Success
            	data["success"] = True

    return jsonify(data)




# Run de application
app.run(host='0.0.0.0',port=port, threaded=False)/home/luis_medina_medina0803/model/cancer_model_full.h5