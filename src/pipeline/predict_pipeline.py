import sys
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from exception import CustomException
from utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):#
        try:

            model_path = 'artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_transformed = preprocessor.transform(features)
            preds = model.predict(data_transformed)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(self,
                 date: str):
        
        self.date = date

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            date_string = self.date
            date_range = pd.date_range(date_string, periods=24, freq='H')
            df = pd.DataFrame(date_range, columns=['Date'])
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

            return df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    print(parent_dir)