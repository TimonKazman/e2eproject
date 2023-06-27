import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

sys.path.append('C:\\Users\\timon\\OneDrive\\Documents\\Data Science\\E2EProject\\src')

from utils import save_object
from exception import CustomException
from logger import logging

'''
    # Get the parent directory of the current file (assuming 'data_transformation.py' is located in the 'components' folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    print(current_dir)
    print(parent_dir)

    # Add the parent directory to the Python path
    sys.path.append(parent_dir)
'''

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def create_features(self, df):

        try:
            df = df.copy()

            logging.info("Transforming data.")

            df = df.set_index("Datetime")
            df.index = pd.to_datetime(df.index)

            logging.info("Adding features.")

            df['hour'] = df.index.hour
            df['dayofweek'] = df.index.dayofweek
            df['quarter'] = df.index.quarter
            df['month'] = df.index.month
            df['year'] = df.index.year
            df['dayofyear'] = df.index.dayofyear
            df['dayofmonth'] = df.index.day
            df['weekofyear'] = df.index.isocalendar().week.astype('int')
            # df['moving_avg_7'] = df['PJME_MW'].rolling(window=7).mean()
            # df['moving_avg_14'] = df['PJME_MW'].rolling(window=14).mean()
            # df['moving_avg_30'] = df['PJME_MW'].rolling(window=30).mean()
            # df['lag_1'] = df['PJME_MW'].shift(364)
            # df['lag_2'] = df['PJME_MW'].shift(728)

            df.dropna(inplace=True)

            return df
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:

            features = ['hour', 'dayofweek', 'quarter', 'month', 'year',
                        'dayofyear']
            target = "PJME_MW"

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            logging.info("Transforming and Feature adding.")

            train_arr = self.create_features(train_df)
            test_arr = self.create_features(test_df)

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj_train = train_arr,
                obj_test = test_arr
                )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
    
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    print("Hello world")
