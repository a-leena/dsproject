import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
# dataclass is a decorator
# used to create class variables without having to define init function
# useful when a class only has variables

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        # variable that will store the 3 paths in the config class
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component.")
        try:
            # it doesn't have to be a csv file, can be any type of database we're reading from
            df = pd.read_csv(r"C:\\Users\\aleen\\OneDrive\\Desktop\\Work\\Data Analyst\\Portfolio\\dsproject\\notebook\\data\\Preprocessed_StudentsPerformance.csv")
            logging.info("Read the dataset as dataframe.")

            # creating the 'artifact' directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            # add artifacts in gitignore so that it doesn't get saved

            # saving the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-Test Split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # saving train data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # saving test data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed.")

            return (self.ingestion_config.train_data_path, 
                    self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__=='__main__':
    obj = DataIngestion()
    train, test = obj.initiate_data_ingestion()
    data_trans = DataTransformation()
    data_trans.initiate_data_transformation(train, test)