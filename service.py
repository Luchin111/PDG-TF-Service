# Developed by Luis H. Medina M. luisinmedina@gmail.com
#Reference: https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037

import pandas as pd
import cv2  as cv
from image_similarity_measures.quality_metrics import rmse, ssim, sre
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
	return '┬íServicio REST activo!'

@app.route('/model/compare/', methods=['GET','POST'])
@cross_origin()
def compare():
    data = {"success": False}
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

       

        image_to_predict = image.load_img(filename, target_size=(224, 224))
        test_image = image.img_to_array(image_to_predict)
        test_image = np.expand_dims(test_image, axis = 0)

        image_to_predict2 = image.load_img(filename2, target_size=(224, 224))
        test_image2 = image.img_to_array(image_to_predict2)
        test_image2 = np.expand_dims(test_image2, axis = 0)

        res = cv.absdiff(test_image, test_image2)
        #print("\ res :",res)
        #Proporci├│n de se├▒al a error de reconstrucci├│n
        res2=sre(test_image, test_image2)
        #Distancia media cuadr├ítica m├şnima
        res3=rmse(test_image, test_image2)
        print("\ res sre :",res2)
        print("\ res rmse :",res3)

        res = res.astype(np.uint8)
        percentage = (np.count_nonzero(res) * 100)/ res.size
        print("\ percentage :",percentage)

        a=(percentage+res2+res3)/3
        b=(percentage+res2+res3)/3
        print("\ percentage prom :",a)

        data["compare"] = []
        r = {"label": "Diferencia", 
            "dif": float(percentage),
            "dif sre": float(res2),
            "dif rmse": float(res3),
            "dif_prom": float(a)}
        
        data["compare"].append(r)

        #Success
        data["success"] = True
        

    return jsonify(data)


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
            	prediction = 1 if (result >= 0.69) else 0
            	CLASSES = ['Normal', 'Cancer']

            	ClassPred = CLASSES[prediction]
            	ClassProb = result
            	
            	print("Predicci├│n:", ClassPred)
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