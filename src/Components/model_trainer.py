import os, sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
# from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Spliting training and testing data")
            
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            
            models_dict = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Linear Regression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor()
            }
            
            # Get the evaluation report for the models
            best_model_name, best_score, evaluation_report = evaluate_models(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, models_dict= models_dict)
            
            #Evaluation report has the model name - key from the models dictionary. We need to extract the model
            best_model = models_dict[best_model_name] 

            if best_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            
        except Exception as e:
            raise CustomException(e,sys)
