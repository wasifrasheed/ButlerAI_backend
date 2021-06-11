import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request
from flask_restful import Api, Resource
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
import sklearn
import os

df=pd.DataFrame()
data=pd.DataFrame()
my_df=pd.DataFrame()
dic=dict()
cd = os.getcwd()
print("CURRENT DIR",cd)

def get_file(my_file,filename):
    global df
    try:
        file_rec = my_file
      
        print("in side get file",file_rec)
        #file_path = f"r'{file_rec}'"
        #print(file_path)
        df = pd.read_csv(cd+"/blue/api/tmp/"+filename,encoding='latin1')
        print(df.head())
        hold_data()
        retJson = {"status":200,"msg":"ok"}
        return retJson
    except Exception as e:
        print("Exception",str(e))
        try:
            # file_rec = request.files['file']
            # print(type(file_rec))
            
            df = pd.read_excel(cd+"/blue/api/tmp/"+filename)
            print("after")
            print(df.head())
            hold_data()
            retJson = {"status":200,"msg":"ok"}
            return retJson
        except Exception as e:
            retJson = {"status":301,"msg":"This file format is not supported"}
            return retJson

def hold_data():
    global data
    data=df.copy()
    print(data.head())
    return data



# preprocessing funtions
def Pre_processing(df,listOfobjColumnNames):
    for name in listOfobjColumnNames:
        values=df[name]
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        df[name]=integer_encoded
    return df


def Scaling(X,y):
    global scalerX,scalery
    scalerX = StandardScaler().fit(X)
    scalery = StandardScaler().fit(y)
    
    X_scale = scalerX.transform(X)
    y_scale = scalery.transform(y)
    return X_scale,y_scale

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
    
    





def drop(drop_list):
    for name in drop_list:
        
        data.drop(name, axis=1,inplace=True)

    return data

def make_target(target,df):
    y=df[target].values
 
    return y



def col_list_send(X):
    col=X.columns
    ls=list()
    for i in range(len(col)):
        ls.append(col[i])
      
    return ls
    
class Models():
    
    global seed 
    seed = 34234

    # Initialization 
    def __init__(self, X,y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
    
        
    # Linear Regression 
    def linear_regression(self):
        reg = LinearRegression()
        reg.fit(self.x_train, self.y_train)
        y_pred_list = reg.predict(self.x_test)
        mse = mean_squared_error(self.y_test, y_pred_list)
        return  mse,reg
        
    # Random Forest Regression model 
    def random_forest(self):
        rfr = RandomForestRegressor(n_estimators=8, max_depth=8, random_state=12, verbose=0)
        rfr.fit(self.x_train, self.y_train)
        y_pred_list = rfr.predict(self.x_test)
        mse = mean_squared_error(self.y_test, y_pred_list)
        return  mse,rfr
            
    # Lasso method 
    def lasso(self):
        reg = Lasso(alpha = 0.1)
        reg.fit(self.x_train, self.y_train)
        y_pred_list = reg.predict(self.x_test)
        mse = mean_squared_error(self.y_test, y_pred_list)

        return  mse,reg
    
    # Gradient Boosing Regressor
    def GBR(self):
        gbr = GradientBoostingRegressor(n_estimators=175, learning_rate=0.08, max_depth=3, random_state=1232, loss='ls')
        gbr.fit(self.x_train, self.y_train)
        mse = mean_squared_error(self.y_test, gbr.predict(self.x_test))
        return  mse,gbr
    
    def Knn_Reg(self):
        knn =KNeighborsRegressor()
        knn.fit(self.x_train, self.y_train)
        mse = mean_squared_error(self.y_test, knn.predict(self.x_test))
        return  mse,knn
    
    def Linear_SVM(self):
        svm =SVR(kernel='linear')
        svm.fit(self.x_train, self.y_train)
        mse = mean_squared_error(self.y_test, svm.predict(self.x_test))
        return  mse,svm
    
    def RBF_SVM(self):
        svm =SVR(kernel='rbf')
        svm.fit(self.x_train, self.y_train)
        mse = mean_squared_error(self.y_test, svm.predict(self.x_test))
        return  mse,svm


def bigno(list_values):
    for num in list_values:
        if num>1:
            return True
    return False
def  acc_update(dic):
    b=dic.values()
    list_values=[]
    for num in b:
        list_values.append(num)
    rescale_current_reading =sklearn.preprocessing.normalize([list_values], norm='l2', axis=1, copy=True, return_norm=False)
    list_acc=[]
    if bigno(list_values)==True:
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


def Main_proceesing(drop_list,target):
    global dic
    global target_y
    global my_df

    try:
        target_y=target
        rec_data=drop(drop_list)
        
        my_df=rec_data.copy()
        
        df_clean=Data_cleaning(rec_data)
    

        
        X=df_clean.drop(target,axis=1)
        y=make_target(target,df_clean)
        col_name=col_list_send(X)

        

        X_scale,y_scale=Scaling(X,y.reshape(-1,1))
        print("shapes")
        print(X_scale)
        print(y_scale)
        models=Models(X_scale,y_scale)
        
        LinearReg_mse,linearReg=models.linear_regression()
        
        RbfSvm_mse,RbfSvm=models.RBF_SVM()
        
        linearSVM_mse,linearSVM=models.Linear_SVM()
    
        Gbr_mse,Gbr=models.GBR()
        Lasso_mse,lasso=models.lasso()
        
        randomforest_mse,randomforest=models.random_forest()
        

        dic={LinearReg_mse:[linearReg,"Linear regression"]
        ,RbfSvm_mse:[RbfSvm,"RBF SVM"]
        ,linearSVM_mse:[linearSVM,"Linear SVM"],
        Gbr_mse:[Gbr,"Gradient boosting regressor"],
        Lasso_mse:[lasso,"Lasso regressor"],
        randomforest_mse:[randomforest,"Random forest regressor"]}
    except Exception as e:
        dic2={"error":"your data shape is not correct","msg":str(e)}
        return dic2
    dic=dict(sorted(dic.items()))
    new_dict=dict()
    n=0
    for k,v in dic.items():

        if n<3:
     
            
            new_dict[k]=v[1]

        n=n+1
    inverse_dic=dict([(value, key) for key, value in new_dict.items()])
    update_dic=acc_update(inverse_dic)
    update_dic["column_name"]=col_name
    print("update dic")
    print(update_dic)
    return update_dic

#prediction Of Regression

def find_model(model_name):
    l=list(dic.values())
    for i in range(len(l)):
        temp=l[i][1]
        if model_name==temp:
            break
    
    return l[i][0],my_df


def Prediction(predict_col,model_name):
    try:
        model,my_df=find_model(model_name)
        print(model)
        my_df=my_df.append(predict_col, ignore_index=True)


        X2=Data_cleaning(my_df)


        X3=X2.drop(target_y,axis=1)
        print("X3",X3)
        X3=scalerX.transform(X3)
        

        prediction=model.predict([X3[-1]])
        print("my prediction")
        print(prediction)
        y_new_inverse = scalery.inverse_transform(prediction)
        return y_new_inverse
    except Exception as e:
        dic2={"error":"Load data try again","msg":str(e)}
        return dic2


