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

@dataclass
class ModelEvaluatorConfig:
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
    params = {
        "Linear Regression": {
            "fit_intercept" : [True, False]
        },
        "K-Neighbors Regression": {
            "n_neighbors": [3,5,7,10,15,20],
            "weights": ['uniform', 'distance'],
            "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        "Decision Tree": {
            "max_depth": [None, 1, 3, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 3, 5, 7, 10]
        },
        "Random Forest Regression": {
            "max_features": ['sqrt', 'log2', None],
            "n_estimators": [10, 20, 50, 75, 100, 500, 1000]
        },
        "XGB Regression": {
            "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5],
            "n_estimators": [10, 20, 50, 75, 100, 500, 1000]
        },
        "CatBoosting Regression": {
            "iterations": [10, 25, 50, 75, 100],
            "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5]
        },
        "AdaBoost Regression": {
            "loss": ['linear', 'square', 'exponential'],
            "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5],
            "n_estimators": [10, 20, 50, 75, 100, 500, 1000]
        },
        "Gradient Boosting Regression": {
            "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5],
            "criterion": ['friedman_mse', 'squared_error'],
            "n_estimators": [10, 20, 50, 75, 100, 500, 1000],
            "min_samples_leaf": [1, 2, 3, 5, 7, 10]
        }
    }

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluator_config = ModelEvaluatorConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data.")
            X_train, y_train, X_test, y_test = (train_array[:,:-1], 
                                                train_array[:,-1], 
                                                test_array[:,:-1], 
                                                test_array[:,-1])
            
            model_r2_report, model_param_report = evaluate_models(
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test, 
                models=self.model_evaluator_config.models,
                params=self.model_evaluator_config.params)
            
            logging.info("All models trained and evaluated.")
            
            best_model_score = max(sorted(model_r2_report.values()))

            # setting a threshold for the scores
            if best_model_score < 0.6:
                raise CustomException("No best model found.", sys)

            best_model_name = list(model_r2_report.keys())[list(model_r2_report.values()).index(best_model_score)]
            best_model_params = model_param_report[best_model_name]
            best_model = self.model_evaluator_config.models[best_model_name]
            logging.info("Best model found.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )
            logging.info("Best model saved.")

            return best_model_name, best_model_score, best_model_params

        except Exception as e:
            raise CustomException(e, sys)
