import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from helper.data_helper import load_data, save_data
from helper.param_helper import load_params
from dvclive import Live

def split_data(df : pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:    
        train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_set, test_set
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")
    
def getNumericalColumns(df: pd.DataFrame) -> list:
    try:
        return df.select_dtypes('number').columns.tolist()
    except Exception as e:
        raise Exception(f"Error getting numerical columns: {e}")

def scale_data(train_set: pd.DataFrame, test_set: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        numerical_columns = getNumericalColumns(train_set)

        scaler = MinMaxScaler()

        train_set[numerical_columns] = scaler.fit_transform(train_set[numerical_columns])
        test_set[numerical_columns] = scaler.transform(test_set[numerical_columns])

        return train_set, test_set
    except Exception as e:
        raise Exception(f"Error scaling data: {e}")

def drop_unwanted_column(train_set: pd.DataFrame, test_set: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_set = train_set.drop(columns, axis=1)
        test_set = test_set.drop(columns, axis=1)
        return train_set, test_set
    except Exception as e:
        raise Exception(f"Error dropping unwanted columns: {e}")

def encode_columns(train_set: pd.DataFrame, test_set: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_set[label] = train_set[label].replace(['<=50K','>50K'], [0,1])
        test_set[label] = test_set[label].replace(['<=50K','>50K'], [0,1])
        train_set_preprocessed = pd.get_dummies(train_set)
        test_set_preprocessed = pd.get_dummies(test_set)

        return train_set_preprocessed, test_set_preprocessed
    except Exception as e:
        raise Exception(f"Error encoding columns: {e}")


def main():
    try:
        param_path = 'params.yaml'
        stage = 'data_preprocessing'
        params = load_params(param_path)

        random_state = params[stage]['random_state']
        test_size = params[stage]['test_size']
        columns_to_drop = params[stage]['columns_to_drop']
        label = params['data']['label_column']
        
        train_data_path = params['data']['train_data_path']
        test_data_path = params['data']['test_data_path']
        cleaned_data_path = params['data']['cleaned_data_path']
        
        df = load_data(cleaned_data_path)

        train_set, test_set = split_data(df, test_size, random_state)
        train_set_scaled, test_set_scaled = scale_data(train_set, test_set)
        train_set_dropped, test_set_dropped = drop_unwanted_column(train_set_scaled, test_set_scaled, columns_to_drop)
        train_set_preprocessed, test_set_preprocessed = encode_columns(train_set_dropped, test_set_dropped, label)

        processed_path = os.path.join('data','processed')
        os.makedirs(processed_path)

        save_data(train_set_preprocessed, train_data_path)
        save_data(test_set_preprocessed, test_data_path)
    except Exception as e:
        raise Exception(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()