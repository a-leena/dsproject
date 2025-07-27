import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data.")
            X_train, y_train, X_test, y_test = (train_array[:,:-1], 
                                                train_array[:,-1], 
                                                test_array[:,:-1], 
                                                test_array[:,-1])
            
            models = {
                    "Linear Regression": LinearRegression(),
                    "K-Neighbors Regression": KNeighborsRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest Regression": RandomForestRegressor(),
                    "XGB Regression": XGBRegressor(),
                    "CatBoosting Regression": CatBoostRegressor(verbose=False),
                    "AdaBoost Regression": AdaBoostRegressor(),
                    "Gradient Boosting Regression": GradientBoostingRegressor()
                    }
            
            model_report = evaluate_models(X_train=X_train, y_train=y_train, 
                                          X_test=X_test, y_test=y_test, 
                                          models=models)
            
            best_model_score = max(sorted(model_report.values()))

            # setting a threshold for the scores (we can add hyperparameter tuning or something here then)
            if best_model_score < 0.6:
                raise CustomException("No best model found.", sys)

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            logging.info("Best model found.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )
            logging.info("Best model saved.")

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
