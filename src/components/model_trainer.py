import os 
import sys
import json
from dataclasses import dataclass

# Get the parent directory of the current file (assuming 'data_transformation.py' is located in the 'components' folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from exception import CustomException
from logger import logging
from utils import save_object, save_object_single, save_metrics

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array) -> dict:
        try:
            logging.info("Split training and test input data.")

            '''
            X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]
            )
            '''

            features = ['hour', 'dayofweek', 'quarter', 'month', 'year',
                        'dayofyear']
            target = "PJME_MW"

            X_train = train_array[features]
            y_train = train_array[target] 

            X_test = test_array[features]
            y_test = test_array[target]

            parameters = {
                 "max_depth": [3, 6, 10],
                 "num_leaves": [10, 30, 100],
                 "learning_rate": [0.01, 0.05, 0.1],
                 "n_estimators": [100, 500, 1000],
                 "colsample_bytree": [0.3, 0.7, 1]
                 }

            lgb_model_hyper = lgb.LGBMRegressor(boosting_type='gbdt',
                                                objective='regression')

            grid_search = RandomizedSearchCV(estimator=lgb_model_hyper, 
                                             param_distributions=parameters, 
                                             scoring='neg_mean_squared_error')
            
            logging.info("Start fitting the model.")

            grid_search.fit(X_train, y_train, 
                            eval_set=[(X_test, y_test)], 
                            eval_metric='rmse')
            
            logging.info(f"Best hyperparmeters found model created. Best params: {grid_search.best_params_}")

            best_model = lgb.LGBMRegressor(boosting_type='gbdt', 
                              objective='regression',
                              **grid_search.best_params_)

            best_model = best_model.fit(X_train, y_train, 
                                        eval_set=[(X_test, y_test)], 
                                        eval_metric='rmse')
            
            save_object_single(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test, num_iteration=best_model.best_iteration_)

            rmse = mean_squared_error(y_test, predicted, squared=False)
            mae = mean_absolute_error(y_test, predicted)
            mape = mean_absolute_percentage_error(y_test, predicted)
            r2 = r2_score(y_test, predicted)

            save_metrics(rmse=rmse, mae=mae, mape=mape, r2=r2)
            logging.info("Metrics in json saved.")

            return r2, rmse, mae, mape

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = ModelTrainer()
    obj.initiate_model_trainer()