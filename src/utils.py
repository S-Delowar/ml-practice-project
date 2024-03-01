import os, sys
import numpy as np, pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
import dill
import pickle

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys)    


def evaluate_models(X_train, X_test, y_train, y_test, models_dict:dict, params:dict):

    logging.info("Evaluation Starts")
    
    model_list = []
    r2_score_list = []
    for i in range(len(models_dict)):
        model = list(models_dict.values())[i]   
        model_key = list(models_dict.keys())[i]
        param = params[model_key]
        
        gs = GridSearchCV(model, param, cv = 3)
        gs.fit(X_train, y_train)
        
        model.set_params(**gs.best_params_)
                     
        model.fit(X_train, y_train)       
        y_predicted = model.predict(X_test)       
        r2score = r2_score(y_test, y_predicted)
        
        model_list.append(model_key)
        r2_score_list.append(r2score)
    
    evaluation_report = list(zip(model_list, r2_score_list))
    evaluation_report = sorted(evaluation_report, key=lambda x: x[1], reverse=True)
    
    best_model_tuple = evaluation_report[0]
    best_model_name = best_model_tuple[0]
    best_model_score = best_model_tuple[1]
    
    logging.info("Evaluation Report Generated")
    
    return best_model_name, best_model_score, evaluation_report
        
        