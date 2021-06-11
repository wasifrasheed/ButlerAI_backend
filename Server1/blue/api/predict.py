from sklearn.cluster import AgglomerativeClustering

from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.applications.vgg16 import preprocess_input 
import numpy as np

from sklearn.decomposition import PCA

import pickle

import os
from cv2 import cv2

def vgg_f():

    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output) 

    print("VGG Model is loaded")
    return model

def extract_features_f(file, model):

    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)

    print("FEATURES extracted")
    print(features)

    return features

def preprocessing_f(path):

    data = {}
    model=vgg_f()

    os.chdir(path)

    # this list holds all the image filename
    animals = []

    # creates a ScandirIterator aliased as files
    with os.scandir(path) as files:
    # loops through each file in the directory
        for file in files:
            if file.name.endswith('.jpg') or file.name.endswith('.png'):
            # adds only the image files to the flowers list
                animals.append(file.name)

    print("images are read")
    print(animals)

    for animal in animals:
            feat = extract_features_f(animal,model)
            print("shape of feat", feat.shape)
            data[animal] = feat

    print("FEATURES")

    # get a list of the filenames
    filenames = np.array(list(data.keys()))
    
    # get a list of just the features
    feat = np.array(list(data.values()))
    
    # reshape so that there are 210 samples of 4096 vectors
    feat = feat.reshape(-1,4096) 

    print("shape of feat after reshaping  :", feat.shape)

    print("x before PCA ")

    print(feat)
    
    pca = PCA(random_state=22)
    pca.fit(feat)
    x = pca.transform(feat)
    print(x)
    print("X with PCA")

    return x, filenames


def clusters(path):

    print("path of folder", path)

    x,names=preprocessing_f(path)
    print("data is processed")

    if len(x) <=10:
        print(len(x))
        threshold= 100

    elif 50 > len(x) > 11:
        print(len(x))
        threshold= 120

    elif 105 > len(x) > 95:
        print(len(x))
        threshold= 215

    elif 205 > len(x) > 195:
        print(len(x))
        threshold= 290

    elif 850 > len(x) > 750:
        print(len(x))
        threshold= 400

    else:
        print(len(x))
        threshold= 300




    model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward',distance_threshold=threshold)
    model.fit(x)
    labels = list(set(model.labels_))

    path_dict= os.path.dirname(os.path.abspath(__file__))
    

    with open(path_dict+'/url_dict.p', 'rb') as fp:
        data = pickle.load(fp)


    # holds the cluster id and the images { id: [images] }
    groups = {}
    for pic, cluster in zip(names,model.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(pic)
        else:
            groups[cluster].append(pic)


    output= {}

    for key in groups.keys():
        print(key)
        
        output[key]=data[groups[key][0]]

    print("output",output)
    
    

    return output

#lab=clusters(r"C:\Users\fizas\Desktop\fyp\Flask_boilerplate\blue\Projects\animal")

