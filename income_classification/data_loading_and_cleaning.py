import pandas as pd
import numpy as np
import os
import yaml
from helper.data_helper import load_data, save_data
from helper.param_helper import load_params

def clean_data(df : pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.replace('?', None)
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.drop('relationship', axis=1)
        return df
    except Exception as e:
        raise Exception(f"Error cleaning data: {e}")

def main():
    try:
        param_path = 'params.yaml'
        params = load_params(param_path)
        file_path = params['data']['raw_data_path']

        data = load_data(file_path)
        cleaned_data = clean_data(data)

        cleaned_path = os.path.join('data','cleaned')
        os.makedirs(cleaned_path)
        save_data(cleaned_data, os.path.join(cleaned_path, 'adult_cleaned.csv'))

        print("Data loaded and cleaned successfully.")
        print(cleaned_data.head())
    except Exception as e:
        raise Exception(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
