import os
import sys
from src.exception import CustomException

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(object, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        r2_report = {}
        params_report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = list(params.values())[i]

            # random_search = RandomizedSearchCV(
            #     estimator=model,
            #     param_distributions=param,
            #     n_iter=20,
            #     cv=5,
            #     scoring='neg_mean_squared_error',
            #     random_state=42,
            #     n_jobs=-1
            # )
            # random_search.fit(X_train, y_train)
            # best_model = random_search.best_estimator_
            # best_params = random_search.best_params_

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            r2_report[list(models.keys())[i]] = r2
            params_report[list(models.keys())[i]] = best_params
        
        return r2_report, params_report
     
    except Exception as e:
        raise CustomException(e, sys)