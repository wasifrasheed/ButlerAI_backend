from flask import Blueprint,jsonify, request
from flask_restful import Api,Resource
from werkzeug.utils import secure_filename

import sys 
import os
import json

from datetime import date, timedelta
import threading

from cv2 import cv2
import numpy as np

### change the "/" to "////" if you are windows user ###


dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-1]
dirname = "/".join(dirname_list)
print(dirname)
path = dirname + "/api"
print(path)
sys.path.append(path)



mod = Blueprint('api',__name__)
api = Api(mod)

###importing the functions from other files ###

path= os.path.dirname(os.path.abspath(__file__))
            
from upload import uploaddata_f
from predict import clusters

class Create_Project(Resource):
    def post(self):
        try:
            proj_path = dirname + "/Projects"

            postedData=request.get_json()
        
            proj_name=postedData['project_name']

            name_and_path = proj_path + "/" + proj_name

            if not os.path.exists(name_and_path):
           
                os.mkdir(name_and_path)
                print("created project : ", proj_name)
                
                ret={"status":200,"message":"Successfully created project"}         
                return jsonify(ret)

            else:
                print(proj_name, " folder already exists.")
                ret={"status":401,"message":"Project with this name already exist"}         
                return jsonify(ret)
  
        except Exception as e:

            ret={"status":401,"message":"Cannot create this project","Problem":e}         
            return jsonify(ret)

class Uploading_Data(Resource):
    def post(self):
        try:
            proj_path = dirname + "/Projects"

            postedData=request.get_json()
        
            proj_name=postedData['project_name']
            urls=postedData['urls']
        
            project_path = proj_path + "/" + proj_name

            #print("project path :", proj_path)
            #print("project name :", proj_name)
            #print("image url :", urls)

            if os.path.exists(project_path):
                
                retJson = uploaddata_f(project_path,urls)
                return jsonify({"status":200,"message":retJson})

            else:
                retJson = {"status":404,"message":"Project with this name doesn't exist"}
                return jsonify(retJson)

        except Exception as e:
            ret={"status":401,"message":"Cannot Upload","Problem":e}         
            return jsonify(ret)
                
class Make_Clusters(Resource):
    def post(self):
        try:

            proj_path = dirname + "/Projects"
            postedData=request.get_json()
        
            proj_name=postedData['project_name']
        
            DIR_NAME = proj_path + "/" + proj_name

            my_dict = clusters(DIR_NAME)
            print((my_dict))
            print(("Heereeee"),type(my_dict))

            retJson = dict()
            for index,value in enumerate((my_dict)):
                print((index))
                my_key = str((index))
                my_value = my_dict[value]
                print((my_value))
                #my_value = value
                retJson[my_key] = my_value
                
            print((retJson))

            retJson["status"] = 200
            # print(retJson)
            #return jsonify({"status":200,"message":retJson})
            return jsonify(retJson)

        
        except Exception as e:
            ret={"status":401,"message":"cannot make clusters","Problem":e}         
            return jsonify(ret)


api.add_resource(Create_Project,"/create")
api.add_resource(Uploading_Data,"/upload")
api.add_resource(Make_Clusters,"/cluster")
