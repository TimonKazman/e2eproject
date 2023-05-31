import os
import sys
import dill

import numpy as np
import pandas as pd

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