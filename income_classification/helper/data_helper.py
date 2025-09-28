import os
import pandas as pd

def load_data(file_path : str) -> pd.DataFrame:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def save_data(df: pd.DataFrame, output_path: str) -> None:
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        raise Exception(f"Error saving data: {e}")

def split_label(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = df.drop(label, axis=1)
        y = df[label]
        return X, y
    except Exception as e:
        raise Exception(f"Error splitting label: {e}")