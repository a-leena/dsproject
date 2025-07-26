import os
import sys
from src.exception import CustomException

import numpy as np
import pandas as pd
import dill

def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(object, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
