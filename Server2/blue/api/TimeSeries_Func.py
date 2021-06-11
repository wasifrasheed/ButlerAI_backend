from flask import Blueprint,jsonify, request
from flask_restful import Api,Resource
import pandas as pd
from pandas.core.common import flatten
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
from datetime import timedelta
from dateutil import relativedelta
from dateutil.relativedelta import *
from dateutil.parser import parse
from sklearn.metrics import mean_squared_error
import json
import sklearn
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
import os

cd = os.getcwd()
print("CURRENT DIR",cd)

df=pd.DataFrame()
data=pd.DataFrame()
dic=dict()
df_stats=pd.DataFrame()

Multi_MseDic=dict()
Uni_MseDic=dict()
ModelDic=dict()
data_for_predic=None
global_mean=None
global_std=None
INDEX=None
check=None
global_past_history=7
global_future_target=None
col_len=None
datetime_values=pd.DataFrame()



def TS_get_file(my_file,filename):
    global df
    try:
        
        file_rec = my_file
        print("in side get file",file_rec)
        df = pd.read_csv(cd+"/blue/api/tmp/"+filename)
        print("Dataframe",df)
        print("Length of DF",(len(df)))
        hold_data()
        retJson = {"status":200,"msg":"ok"}
        return retJson
    except Exception as e:
        try:
            #file_rec = request.files['file']
           
            df = pd.read_excel(cd+"/blue/api/tmp/"+filename)
        
            hold_data()
            retJson = {"status":200,"msg":"ok"}
            return retJson
        except Exception as e:
            retJson = {"status":301,"msg":"This file format is not supported"}
            return retJson



# preprocessing funtions
def Pre_processing(df,listOfobjColumnNames):
  
    for name in listOfobjColumnNames:
        values=df[name]
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        df[name]=integer_encoded
    return df
    
def Scaling(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def Data_cleaning(df):
    df = df.replace(['?','na'], np.nan)
    
    df[df.select_dtypes(['category']).columns] = df.select_dtypes(['category']).apply(lambda x: x.astype('object'))
    
    df[df.select_dtypes(['datetime64[ns]']).columns] = df.select_dtypes(['datetime64[ns]']).apply(lambda x: x.astype('object'))
    filteredColumns = df.dtypes[df.dtypes != np.object]
    listOfColumnNames = list(filteredColumns.index)
    for name in listOfColumnNames:
        df[name]=df[name].fillna(df[name].mean())

    objectlist = df.dtypes[df.dtypes == np.object]
    listOfobjColumnNames = list(objectlist.index)
    for name in listOfobjColumnNames:
        df[name]=df[name].fillna(df[name].mode()) 
    df=df.dropna()
    
    df= Pre_processing(df,listOfobjColumnNames)
    #df=Scaling(df)
    return df
    

def hold_data():
    global data
    data=df.copy()
    return data


def drop(drop_list):
    for name in drop_list:
        data.drop(name, axis=1,inplace=True)
    return data

def make_target(target,df):
    y=df[target].values
    return y

def Transform(y,mean,std):
        return (y-mean)/std
    
def Inverse_Transform(y,mean,std):
        return (y*std+mean)

def return_datetime(latest_date,future_num,time_differece,time_var):
    DateTime_df=pd.DataFrame()
    col_name=latest_date.name
    datelist=[]
    if time_var=="Years":
        latest_date=latest_date[0]
        for _ in range(future_num):
            latest_date = latest_date+relativedelta(years=+time_differece)
            DateTime_df=DateTime_df.append(pd.DataFrame({latest_date:0}.keys(),columns=[col_name]),ignore_index=True)
        l=DateTime_df[col_name].tolist()
        for a in l:
            datelist.append(str(a))
        return datelist
    
    if time_var=="Months":
        latest_date=latest_date[0]
        for _ in range(future_num):
            latest_date = latest_date+relativedelta(months=+time_differece)
            DateTime_df=DateTime_df.append(pd.DataFrame({latest_date:0}.keys(),columns=[col_name]),ignore_index=True)
        l=DateTime_df[col_name].tolist()
        for a in l:
            datelist.append(str(a))
        return datelist
        
    if time_var=='Days':
        days=time_differece
        hours=0
        minutes=0
    elif time_var=="Hours":
        days=0
        hours=time_differece
        minutes=0
    elif time_var=="Minutes":
        days=0
        hours=0
        minutes=time_differece
        
    for _ in range(future_num):
        latest_date = latest_date + timedelta(hours=hours,days=days,minutes=minutes)
        DateTime_df=DateTime_df.append(pd.DataFrame(latest_date),ignore_index=True)
    l=DateTime_df[col_name].tolist()
    for a in l:
        datelist.append(str(a))
    return datelist
        

def TS_Main_proceesing(drop_list,target,date_col,future,time_diff,time_var):
    global Uni_MseDic
    global Multi_MseDic
    global ModelDic
    global data_for_predic
    global global_mean
    global global_std
    global INDEX
    global check
    global global_future_target
    global col_len
    global df_stats
    global datetime_values
    print(date_col)
    try:
        print("the data set")
        dataset=drop(drop_list)
        #dataset=dataset[:100]
        #dataset=main_date(dataset)
        print(dataset.head())
        print("the data set")
        
        dataset.set_index([date_col],inplace=True,drop = True)
        
        print(dataset)
        try:
            dataset.index = pd.to_datetime(dataset.index)

        except Exception as e:
            retJson = {"status":301,"msg":"DateTime format is not supported"}
            return retJson
        print(dataset)
        latest_date = dataset.iloc[[-1]].index
        dataset=Data_cleaning(dataset)
        df_stats=dataset
        target_col=target
        col_len=len(dataset.columns)
        INDEX=dataset.columns.get_loc(target_col)
        #if col_len==1:
        #   dataset=dataset[target_col].values
        #else:
        #    dataset=dataset.values
        dataset=dataset.values
        print(dataset)
        global_future_target=int(future)
        current_data=dataset[0:global_past_history]
        data_for_predic=current_data
        dataset=dataset[global_past_history:]
        data_mean = dataset.mean(axis=0)
        data_std = dataset.std(axis=0)
        global_mean=data_mean
        global_std=data_std
        datetime_values=return_datetime(latest_date,global_future_target,time_diff,time_var)
        dataset=Transform(dataset,global_mean,global_std)
        data_for_predic=Transform(data_for_predic,global_mean,global_std)

        if col_len>1:
            check="Multi"
            multi_models=TS_Multi_Models(dataset,INDEX,global_future_target,global_past_history)
            Mse_LSTM,Multi_LSTM_model=multi_models.MultiVariate_LSTM()
            Mse_RNN,Multi_RNN_model=multi_models.MultiVariate_RNN()
            Multi_MseDic={Mse_LSTM:"Multi LSTM model",Mse_RNN:"Multi RNN model"}
            ModelDic={"Multi LSTM model":Multi_LSTM_model,"Multi RNN model":Multi_RNN_model}
            Multi_MseDic=dict(sorted(Multi_MseDic.items()))
            inverse_dic=dict([(value, key) for key, value in Multi_MseDic.items()])
            update_dic=acc_update_Multi(inverse_dic)
            return update_dic
        else:
            check="Uni"
            print(dataset)
            uni_models=TS_Uni_Models(dataset,global_future_target,global_past_history)
            Mse_LSTM,Uni_LSTM_model=uni_models.Univariate_LSTM()
            Mse_RNN,Uni_RNN_model=uni_models.Univariate_RNN()
            dic_stats,df_stat=general_univarient(df_stats)
            UniStats_MseDic={k:v[1] for k,v in dic_stats.items()}
            Uni_MseDic={Mse_LSTM:"Uni LSTM model",Mse_RNN:"Uni RNN model"}
            Uni_MseDic.update(UniStats_MseDic)
            ModelDic={"Uni LSTM model":Uni_LSTM_model,"Uni RNN model":Uni_RNN_model}
            Uni_MseDic=dict(sorted(Uni_MseDic.items()))
            new_dict=dict()
            n=0
            for k,v in Uni_MseDic.items():
                if n<3:
                    new_dict[k]=v
                n=n+1
            inverse_dic=dict([(value, key) for key, value in new_dict.items()])
            update_dic=acc_update_Uni(inverse_dic)
            return update_dic
    except Exception as e:
        dic2={"error":"Data is not in required shape","msg":str(e)}
        return dic2

class TS_Multi_Models():
    
    def __init__(self,df,target_col,future_target,past_history):
        self.df=df
        self.future_target=future_target
        self.target_col=target_col
        self.past_history=past_history
    
    def multivariate_data(self,dataset, target, start_index, end_index, history_size,target_size, step, single_step=False):
        data = []
        labels = []
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size
        for i in range(start_index, end_index):
            indices = range(i-history_size, i, step)
            data.append(dataset[indices])
            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])
        return np.array(data), np.array(labels)

    def CalMSE(self,actual,pred):
        return mean_squared_error(actual,pred)

    def MultiVariate_LSTM(self):
        df=self.df
        dataset = df
        len_dataset=len(dataset)
        BATCH_SIZE = 1
        BUFFER_SIZE = 1000
        EVALUATION_INTERVAL =(len_dataset//BATCH_SIZE)
        EPOCHS = 1
        past_history =self.past_history
        future_target = self.future_target
        STEP = 1
        #INDEX=self.target_col
        x_train_multi, y_train_multi = self.multivariate_data(dataset, dataset[:, INDEX], 0,None, past_history,future_target, STEP)
        x_test_multi, y_test_multi =x_train_multi[0:1],y_train_multi[0:1]
        x_train_multi,y_train_multi=x_train_multi[1:],y_train_multi[1:]
        train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
        train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        test_data_multi = tf.data.Dataset.from_tensor_slices((x_test_multi, y_test_multi))
        test_data_multi = test_data_multi.batch(BATCH_SIZE).repeat()
        multi_step_model = tf.keras.models.Sequential()
        multi_step_model.add(tf.keras.layers.LSTM(32,return_sequences=True,input_shape=x_train_multi.shape[-2:]))
        multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
        multi_step_model.add(tf.keras.layers.Dense(future_target))
        multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
        multi_step_model.fit(train_data_multi, epochs=EPOCHS,verbose=1,steps_per_epoch=EVALUATION_INTERVAL)
        prediction=None

        for x, y in test_data_multi.take(1):
            prediction=multi_step_model.predict(x)
            y=y.numpy()
        mse=self.CalMSE(y,prediction)    
        return mse,multi_step_model

    def MultiVariate_RNN(self):
        df=self.df
        dataset = df
        len_dataset=len(dataset)
        BATCH_SIZE = 1
        BUFFER_SIZE = 1000
        EVALUATION_INTERVAL =len_dataset//BATCH_SIZE
        EPOCHS = 1
        past_history =self.past_history
        future_target = self.future_target
        STEP = 1
        #INDEX=self.target_col
        x_train_multi, y_train_multi = self.multivariate_data(dataset, dataset[:, INDEX], 0,None, past_history,future_target, STEP)
        x_test_multi, y_test_multi =x_train_multi[0:1],y_train_multi[0:1]
        x_train_multi,y_train_multi=x_train_multi[1:],y_train_multi[1:]
        train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
        train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        test_data_multi = tf.data.Dataset.from_tensor_slices((x_test_multi, y_test_multi))
        test_data_multi = test_data_multi.batch(BATCH_SIZE).repeat()
        multi_step_model = tf.keras.models.Sequential()
        multi_step_model.add(tf.keras.layers.SimpleRNN(32,return_sequences=True,input_shape=x_train_multi.shape[-2:]))
        multi_step_model.add(tf.keras.layers.SimpleRNN(16, activation='relu'))
        multi_step_model.add(tf.keras.layers.Dense(future_target))
        multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
        multi_step_model.fit(train_data_multi, epochs=EPOCHS,verbose=1,steps_per_epoch=EVALUATION_INTERVAL)
        prediction=None
        
        for x, y in test_data_multi.take(1):
            prediction=multi_step_model.predict(x)
            y=y.numpy()
        mse=self.CalMSE(y,prediction)
        return mse,multi_step_model


class TS_Uni_Models():

    def __init__(self,df,future_target,past_history):
        self.df=df
        self.future_target=future_target
        self.past_history=past_history

    def univariate_data(self,dataset, start_index, end_index, history_size, target_size):
        data = []
        labels = []
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size
        for i in range(start_index, end_index):
            indices = range(i-history_size, i)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i+target_size])
        return np.array(data), np.array(labels)
    
    def CalMSE(self,actual,pred):
        return mean_squared_error(actual,pred)
    
    def Univariate_LSTM(self):
        df=self.df
        uni_data = df
        len_uni_data=len(uni_data)
        BATCH_SIZE = 1
        BUFFER_SIZE = 1
        EVALUATION_INTERVAL =len_uni_data//BATCH_SIZE
        EPOCHS = 1
        univariate_past_history=self.past_history
        univariate_future_target = self.future_target
        STEP = 1
       
        x_train_uni, y_train_uni = self.univariate_data(uni_data, 0, None,univariate_past_history,univariate_future_target)
        x_val_uni, y_val_uni =x_train_uni[0:1],y_train_uni[0:1]
        x_train_uni,y_train_uni=x_train_uni[1:],y_train_uni[1:]
        train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
        train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
        val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
        simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32,return_sequences=True,input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.LSTM(8,activation='relu'),
        tf.keras.layers.Dense(1)
        ])
        simple_lstm_model.compile(optimizer='adam', loss='mae')
        simple_lstm_model.fit(train_univariate, epochs=EPOCHS,steps_per_epoch=EVALUATION_INTERVAL)
        prediction=None
        for x, y in val_univariate.take(1):
            prediction=simple_lstm_model.predict(x)
            y=y.numpy()  
        mse=self.CalMSE(y,prediction)
        return mse,simple_lstm_model
    
    def Univariate_RNN(self):
        df=self.df
        uni_data = df
        len_uni_data=len(uni_data)
        BATCH_SIZE = 1
        BUFFER_SIZE = 1
        EVALUATION_INTERVAL =len_uni_data//BATCH_SIZE
        EPOCHS = 1
        univariate_past_history=self.past_history
        univariate_future_target = self.future_target
        STEP = 1

        x_train_uni, y_train_uni = self.univariate_data(uni_data, 0, None,univariate_past_history,univariate_future_target)
        x_val_uni, y_val_uni =x_train_uni[0:1],y_train_uni[0:1]
        x_train_uni,y_train_uni=x_train_uni[1:],y_train_uni[1:]                       
        train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
        train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
        val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
        simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(32,return_sequences=True,input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.SimpleRNN(8,activation='relu'),
        tf.keras.layers.Dense(1)
        ])
        simple_lstm_model.compile(optimizer='adam', loss='mae')
        simple_lstm_model.fit(train_univariate, epochs=EPOCHS,steps_per_epoch=EVALUATION_INTERVAL)
        prediction=None
        for x, y in val_univariate.take(1):
            prediction=simple_lstm_model.predict(x)
            y=y.numpy()
        mse=self.CalMSE(y,prediction)
        return mse,simple_lstm_model

##STATS MODELS

def general_univarient(df):
    #df,df_o=Collect_Data(df)
    col_name=df.columns[0]
    train,test=split_data(df)
    model_ar,r2_ar=AutoReg_Prediction_Testing(df,train,test,col_name)
    model_sarimax,r2_sarimax=Sarimax_model(df,train,test,col_name)
    model_arima,r2_arima=Arima_model(df,train,test,col_name)
    model_arma,r2_arma=ARMA_model(df,train,test,col_name)
    dic={r2_ar:[model_ar,"AR"],r2_sarimax:[model_sarimax,"SARIMAX"],r2_arima:[model_arima,"ARIMA"],
    r2_arma:[model_arma,"ARMA"]}
    dic=dict(sorted(dic.items()))
    return dic,df

def AutoReg_Prediction_Testing(df,train,test,col_name):
    model = AR(train)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test)-1, dynamic=False)
    length_of_test = len(test)
    compare_df = pd.concat(
    [df[col_name].tail(length_of_test),
    predictions], axis=1).rename(
    columns={'stationary': 'actual', 0:'predicted'})
    r2 = mean_squared_error(df[col_name].tail(length_of_test), predictions.tail(length_of_test))
    return model_fit,r2
                
def Arima_model(df,train,test,col_name):
    model = ARIMA(train, order=(1, 1, 1))
    model_fit_arima = model.fit(disp=False)
    predictions_arima = model_fit_arima.predict(start=len(train), end=len(train) + len(test)-1, dynamic=False)
    length_of_test = len(test)
    compare_df_arima = pd.concat(
    [df[col_name].tail(length_of_test),
    predictions_arima], axis=1).rename(
    columns={'stationary': 'actual', 0:'predicted'})
    r2_arima = mean_squared_error(df[col_name].tail(length_of_test), predictions_arima.tail(length_of_test))
    return model_fit_arima,r2_arima

def Sarimax_model(df,train,test,col_name):
    model = SARIMAX(train, order=(1, 1, 1))
    model_fit_sarima = model.fit(disp=False)
    length_of_test = len(test)
    predictions_sarima = model_fit_sarima.predict(start=len(train), end=len(train) + len(test)-1, dynamic=False)
    compare_df_sarima = pd.concat(
    [df[col_name].tail(length_of_test),
    predictions_sarima], axis=1).rename(
    columns={'stationary': 'actual', 0:'predicted'})
    r2_sarima = mean_squared_error(df[col_name].tail(length_of_test), predictions_sarima.tail(length_of_test))##changes
    return model_fit_sarima,r2_sarima

def ARMA_model(df,train,test,col_name):
    model = ARMA(train, order=(0, 1))
    model_fit_arma = model.fit(disp=False)
    length_of_test = len(test)
    predictions_arma = model_fit_arma.predict(start=len(train), end=len(train) + len(test)-1, dynamic=False)
    compare_df_arma = pd.concat(
    [df[col_name].tail(length_of_test),
    predictions_arma], axis=1).rename(
    columns={'stationary': 'actual', 0:'predicted'})
    r2_arma = mean_squared_error(df[col_name].tail(length_of_test), predictions_arma.tail(length_of_test))
    return model_fit_arma,r2_arma

def AutoReg_Prediction(df,col):
    model = AR(df)
    model_fit = model.fit()
    yhat = model_fit.predict(len(df),len(df))
    s = pd.Series(yhat,name=col)
    df2=s.to_frame()
    df=df.append(df2)
    return df

def ARMA_Prediction(df,col):
    model = ARMA(df, order=(0, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(df), len(df))
    s = pd.Series(yhat,name=col)
    df2=s.to_frame()
    df=df.append(df2)
    return df

def SARIMA_Prediction(df,col):
    model = SARIMAX(df, order=(1, 1, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(df), len(df))
    s = pd.Series(yhat,name=col)
    df2=s.to_frame()
    df=df.append(df2)
    return df

def ARIMA_Prediction(df,col):
    model = ARIMA(df, order=(1, 1, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(df), len(df), typ='levels')
    s = pd.Series(yhat,name=col)
    df2=s.to_frame()
    df=df.append(df2)
    return df

def Collect_Data(df):
    num=date_chk(df)
    df=change_index(df,num)
    orignal_df = df.copy()
    return df,orignal_df

def split_data(df):
    split=len(df)*0.8
    split=int(split)
    train=df[:split]
    test=df[split:]
    return train,test

def main_prediction(model_name,df,no_pri):
    col_name=df.columns[0]
    if model_name=="ARIMA":
        for i in range(100):
            df=ARIMA_Prediction(df,col_name)
        return(df[:no_pri])
    elif model_name=="AR":
        for i in range(100):
            df=AutoReg_Prediction(df,col_name)
        return(df[:no_pri])
    elif model_name=="ARMA":
        for i in range(100):
            df=ARMA_Prediction(df,col_name)
        return(df[:no_pri])
    elif model_name=="SARIMAX":
        for i in range(100):
            df=SARIMA_Prediction(df,col_name)
        return(df[:no_pri])
        

## End Stats Models

def find_model(model_name):
    l=list(ModelDic.keys())
    for i in range(len(l)):
        temp=l[i]
        if model_name==temp:
            return ModelDic[model_name]

def TS_Prediction(model_name):
    try:
        model=find_model(model_name)
        if check=="Multi":
            data_for_predict=np.reshape(data_for_predic,(1,global_past_history,col_len))
            prediction=model.predict(data_for_predict)
            prediction=Inverse_Transform(prediction,global_mean[INDEX],global_std[INDEX])
            prediction=prediction.tolist()
            return {"DateTime":datetime_values,"Predictions":prediction[0]}
        elif check=="Uni":
            if (model_name=="Uni RNN model") or (model_name=="Uni LSTM model"):
                prediction=make_forecast(model,data_for_predic,global_past_history,global_future_target)
                return {"DateTime":datetime_values,"Predictions":prediction}
            else:
                statsPred= main_prediction(model_name,df_stats,global_future_target)
                statsPred=statsPred.values
                pList=[]
                for i in statsPred:
                    pList.append(i[0])
                
                return {"DateTime":datetime_values,"Predictions":pList}
    except Exception as e:
        dic2={"error":"Load data try again","msg":str(e)}
        return dic2

def make_forecast(model,current_data,past,future):
    pred_list=[]
    scaled_pred_list=[]
    for _ in range(0,future):
        data_forecast=np.reshape(current_data,(1,past,1))
        pred=model.predict(data_forecast)
        pred_list.append(pred)
        current_data=np.insert(current_data, 0, pred)
        current_data=current_data[0:past]
    pred_list=list(flatten(pred_list))
    for f in pred_list:
        scaled_pred=Inverse_Transform(f,global_mean[INDEX],global_std[INDEX])
        scaled_pred_list.append(scaled_pred)
    return scaled_pred_list
       

    
    



def bigno_Uni(list_values):
    for num in list_values:
        if num>1:
            return True
    return False
def  acc_update_Uni(dic):
    b=dic.values()
    list_values=[]
    for num in b:
        list_values.append(num)
    rescale_current_reading =sklearn.preprocessing.normalize([list_values], norm='l2', axis=1, copy=True, return_norm=False)
    list_acc=[]
    if bigno_Uni(list_values)==True:
        for i in range(3):
            acc=(1-rescale_current_reading[0][i])*100
            list_acc.append(acc)
    else:
        for i in range(3):
            acc=(1-list_values[i])*100
            list_acc.append(acc)
    new_dic=dict()
    d=dic.keys()
    key_list=[]
    for num in d:
        key_list.append(num)
    for i in range(3):
        key=key_list[i]
        acc=list_acc[i]
        error=rescale_current_reading[0][i]
        new_dic[key]=[acc,error]
    return new_dic


def bigno_Multi(list_values):
    for num in list_values:
        if num>1:
            return True
    return False

def  acc_update_Multi(dic):
    b=dic.values()
    list_values=[]
    for num in b:
        list_values.append(num)
    rescale_current_reading =sklearn.preprocessing.normalize([list_values], norm='l2', axis=1, copy=True, return_norm=False)
    list_acc=[]
    if bigno_Multi(list_values)==True:
        for i in range(2):
            acc=(1-rescale_current_reading[0][i])*100
            list_acc.append(acc)
    else:
        for i in range(2):
            acc=(1-list_values[i])*100
            list_acc.append(acc)
    new_dic=dict()
    d=dic.keys()
    key_list=[]
    for num in d:
        key_list.append(num)
    for i in range(2):
        key=key_list[i]
        acc=list_acc[i]
        error=rescale_current_reading[0][i]
        new_dic[key]=[acc,error]
    return new_dic
