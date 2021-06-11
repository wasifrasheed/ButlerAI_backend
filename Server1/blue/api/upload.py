from flask import Blueprint,jsonify, request
from flask_restful import Api,Resource
import pandas as pd
import os
from PIL import Image
from io import BytesIO
import requests
from PIL import Image
import urllib
import numpy as np
import pickle
import os

def uploaddata_f(projectpath,urls):

    url_dict={}
    path= os.path.dirname(os.path.abspath(__file__))
    print(path)

    for url in urls:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        
        imgarr = np.array(img) 

        #print(imgarr)
        #filename = url.split('.')[-2]
        filename = url.split('/')[-1]

        url_dict[filename]=url

        name = filename.split('.')[-2]
        #print(name)
        ext = filename.split('.')[-1]
        #print(ext)
        fn = name + '.' + ext 
        image = Image.fromarray(imgarr, 'RGB')
        location = os.path.join(projectpath, fn)
        image.save(location)

    with open(path+'/url_dict.p', 'wb') as fp:
        pickle.dump(url_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return "Successfully Uploaded"








'''
    #loading data
    # change the working directory to the path where the images are located
    os.chdir(path)

    # this list holds all the image filename
    animals = []

    # creates a ScandirIterator aliased as files
    with os.scandir(path) as files:
    # loops through each file in the directory
        for file in files:
            if file.name.endswith('.jpg') or file.name.endswith('.png') : 
            # adds only the image files to the animal list
                animals.append(file.name)
        
    print("LOAD DATA")
    return animals
    '''