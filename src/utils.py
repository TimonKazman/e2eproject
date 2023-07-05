import os
import sys
import dill
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

sys.path.append('C:\\Users\\timon\\OneDrive\\Documents\\Data Science\\E2EProject\\src')
from exception import CustomException

def save_object(file_path, obj_train=None, obj_test=None):

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump((obj_train, obj_test), file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def save_object_single(file_path, obj=None):

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
# not used in script ...
def evaluate_models(X_train, y_train, X_test, y_test, models, param) -> dict:
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
def save_metrics(rmse=None, mae=None, mape=None, r2=None):
    try:

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2
        }

        folder_path = 'metrics'
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, 'metrics.json')
        with open(file_path, 'w') as file:
            json.dump(metrics, file)

    except Exception as e:
        raise CustomException(e, sys)


    