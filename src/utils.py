import os, sys
import numpy as np, pandas as pd

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
import dill

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    
    except Exception as e:
        raise CustomException(e, sys)
        


def evaluate_models(X_train, X_test, y_train, y_test, models_dict:dict):

    logging.info("Evaluation Starts")
    
    model_list = []
    r2_score_list = []
    for i in range(len(models_dict)):
        model_name = list(models_dict.keys())[i]
        model = list(models_dict.values())[i]        
        model.fit(X_train, y_train)       
        y_predicted = model.predict(X_test)       
        r2score = r2_score(y_test, y_predicted)
        
        model_list.append(model_name)
        r2_score_list.append(r2score)
    
    evaluation_report = list(zip(model_list, r2_score_list))
    evaluation_report = sorted(evaluation_report, key=lambda x: x[1], reverse=True)
    
    best_model_tuple = evaluation_report[0]
    best_model_name = best_model_tuple[0]
    best_model_score = best_model_tuple[1]
    
    logging.info("Evaluation Report Generated")
    
    return best_model_name, best_model_score, evaluation_report
        
        