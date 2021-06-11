from flask import Blueprint,jsonify, request
from flask_restful import Api,Resource
import pandas as pd
from werkzeug.utils import secure_filename
import sys 
import os
import json

from blue import app
import logging

# logging.basicConfig(filename='demo.log',level=logging.DEBUG,
# format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s")

dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-1]
dirname = "/".join(dirname_list)
print(dirname)
path = dirname + "/api"
print(path)
sys.path.append(path)



from Functions import get_file
from Functions import Main_proceesing,Prediction
from TimeSeries_Func import TS_Main_proceesing,TS_Prediction,TS_get_file

mod = Blueprint('api',__name__)
api = Api(mod)
df = None





path01=dirname +"/api/tmp"

class data_for_Regression(Resource):
    def post(self):
        #app.logger.info('Processing default request')
        file_rec = request.files['file']
        print(file_rec)
        # df = pd.read_csv(file_rec,encoding= 'unicode_escape')
        # print(df.head())

        filename = secure_filename(file_rec.filename) 
        print(filename)
        print(path01)
        file_path =  path01 +"/" + filename
        file_rec.save(file_path)
        print((f"File saved! on {file_path}"))

        #df.to_csv(file_path,index=False)
        get_file(file_path,filename)

        postedData=json.loads(request.form.get('json'))
        delete=postedData["drop"]
        target=postedData['target']
        ret=Main_proceesing(delete,target)
        #os.remove(file_path)
        return jsonify(ret)
  

class Reg_Prediction(Resource):
    def post(self):
        app.logger.info('Processing default request')
        postedData=request.get_json()
        print(postedData)
        col_name=postedData['columns']
        print(col_name)
        name_model=postedData['model']
        ret=Prediction(col_name,name_model)
        print("done")
        ret=str(ret[0])
        return jsonify(ret)


class data_for_timeseries(Resource):
    def post(self):
        app.logger.info('Processing default request')
        file_rec = request.files['file']
        # df = pd.read_csv(file_rec)
        # print(df.head())

        filename = secure_filename(file_rec.filename) 
        print(filename)
        print(path01)
        file_path =  path01 +"/" + filename
        file_rec.save(file_path)
        #df.to_csv(file_path,index=False)
        
        TS_get_file(file_path,filename)
        
        postedData=json.loads(request.form.get('json'))
        delete=postedData["drop"]
        target=postedData['target']
        future=postedData['future']
        time_diff=postedData['time_diff']
        time_var=postedData['time_var']
        date_col=postedData['date_col']
        ret=TS_Main_proceesing(delete,target,date_col,future,time_diff,time_var)
        return jsonify(ret)


class TimeSeries_Prediction(Resource):
    def post(self):
        app.logger.info('Processing default request')
        postedData=request.get_json()
        name_model=postedData['model']
        ret=TS_Prediction(name_model)
        #ret=str(ret)
        return jsonify(ret)


api.add_resource(data_for_Regression, "/data_reg")
api.add_resource(Reg_Prediction, "/prediction")
api.add_resource(data_for_timeseries,"/data_TS")
api.add_resource(TimeSeries_Prediction,"/TS_prediction")
