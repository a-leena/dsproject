import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    # if we want to save any preprocessor model as a pickle file (also applicable for other models)
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            numerical_cols = ['math_score', 'reading_score', 'writing_score']
            categorical_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            # we will combine in this pipeline processes like handling missing values with imputation and applying standard scaler
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Created Numerical pipeline.")

            # combine in this pipeline the process of handling missing values, converting categorical values to numerical, and applying standard scaler to those numerical values
            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            # using standard scaler here isn't necessary but we can
            # but applying onehot encoder gives us a sparse matrix
            # standard scaler would subtract from mean by default
            # which makes the sparse matrix into dense matrix and such a process isn't allowed
            # so we can either forego the standard scaling or put a parameter 'with_mean=False'
            # this tells the scaler to not subtract the mean    
            logging.info("Created Categorical pipeline.")

            preprocessor = ColumnTransformer(
                [
                    ("categorical_pipeline", categorical_pipeline, categorical_cols),
                    ("numerical_pipeline", numerical_pipeline, numerical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test data.")

            logging.info("Obtaining preprocessor object.")
            preprocessor = self.get_data_transformer_obj()

            target_column_name = 'average_score'

            input_feature_train_df = train_df.drop(columns=[target_column_name, 'total_score'], axis=1)
            target_feature_train_df = train_df[target_column_name]
            print(input_feature_train_df)

            input_feature_test_df = test_df.drop(columns=[target_column_name, 'total_score'], axis=1)
            target_feature_test_df = test_df[target_column_name]
            print(input_feature_test_df)

            logging.info("Applying preprocessor object on training and testing dataframes.")
            input_feature_train_array = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor.transform(input_feature_test_df) # we do transform using the preprocessor model that has been fit to the train dataset above

            # combining the i/p and target features into a single array
            train_array = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                object = preprocessor
            )
            logging.info("Saved preproccesed object.")
            
            return (train_array, 
                    test_array, 
                    self.data_transformation_config.preprocessor_obj_file_path)





        except Exception as e:
            raise CustomException(e, sys)