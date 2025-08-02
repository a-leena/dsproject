import sys
from dataclasses import dataclass
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

@dataclass
class PredictionConfig:
    preprocessor_path = 'artifacts/preprocessor.pkl'
    model_path = 'artifacts/model.pkl'

class PredictPipeline:
    def __init__(self, preprocessor_path, model_path):
        # model_path = 'artifacts\model.pkl'
        # preprocessor_path = 'artifacts\preprocessor.pkl'
        self.preprocessor = load_object(file_path = preprocessor_path)
        self.model = load_object(file_path = model_path)

    def predict(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            logging.info("Data is preprocessed.")
            prediction = self.model.predict(data_scaled)
            logging.info("Prediction is made.")
            return prediction

        except Exception as e:
            raise CustomException(e, sys)

# maps the data given as input in the webpage to the backend
class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_dataframe(self):
        try:
            data_input = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            # print(data_input)
            logging.info("Obtained user inputs as dataframe.")
            return pd.DataFrame(data_input)
        
        except Exception as e:
            raise CustomException(e, sys)