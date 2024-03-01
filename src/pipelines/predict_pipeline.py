import os, sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object



class PredictPipeline:
    def __init__(self) -> None:
        pass
        
    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print('before loading')
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            print('after loading')
            
            data_processed = preprocessor.transform(features)
            pred = model.predict(data_processed)
            
            return pred
                       
        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_dataframe(self):
        try:
            custom_data_dict = {
                "gender" : [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            
            custom_dataframe = pd.DataFrame(custom_data_dict)
            return custom_dataframe
        
        except Exception as e:
            raise CustomException(e, sys)
        