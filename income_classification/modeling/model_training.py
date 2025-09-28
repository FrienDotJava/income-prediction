from sklearn.ensemble import GradientBoostingClassifier
from helper.data_helper import load_data, split_label
from helper.param_helper import load_params
import pandas as pd
import pickle

def initiate_model(n_estimators : int, max_depth : int, learning_rate : float) -> GradientBoostingClassifier:
    try:
        return GradientBoostingClassifier(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate)
    except Exception as e:
        raise Exception(f"Error initiating model: {e}")

def train_model(model: GradientBoostingClassifier, X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise Exception(f"Error training model: {e}")

def save_model(model : GradientBoostingClassifier, save_path: str) -> None:
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        raise Exception(f"Error saving model: {e}")

def main():
    try:
        param_path = 'params.yaml'
        stage = 'model_training'
        params = load_params(param_path)

        train_data_path = params['data']['train_data_path']
        model_path = params['model_path']

        n_estimators = params[stage]['n_estimators']
        max_depth = params[stage]['max_depth']
        learning_rate = params[stage]['learning_rate']
        label = params['data']['label_column']

        train_data = load_data(train_data_path)

        X_train, y_train = split_label(train_data, label)

        model = initiate_model(n_estimators, max_depth, learning_rate)

        trained_model = train_model(model, X_train, y_train)

        save_model(trained_model, model_path)
    except Exception as e:
        raise Exception(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
    




