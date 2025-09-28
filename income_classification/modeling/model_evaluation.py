import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from income_classification.helper.data_helper import load_data, split_label
from income_classification.helper.param_helper import load_params
from sklearn.ensemble import GradientBoostingClassifier


def load_model(path : str) -> GradientBoostingClassifier:
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(f"Error loading model: {e}")
    
def predict(model : GradientBoostingClassifier, X_test : pd.DataFrame) -> pd.Series:
    try:
        y_pred = model.predict(X_test)
        return y_pred
    except Exception as e:
        raise Exception(f"Error making predictions: {e}")

def evaluate(y_test : pd.Series, y_pred : pd.Series) -> tuple[float, float, float, float]:
    try:
        acc = accuracy_score(y_test, y_pred)
        pres = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return acc, pres, recall, f1
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")

def to_dict(acc : float, pres : float, recall : float, f1 : float) -> dict:
    try:
        return {
            'accuracy':acc,
            'precision': pres,
            'recall':recall,
            'f1_score':f1
        }
    except Exception as e:  
        raise Exception(f"Error converting metrics to dict: {e}")

def save_metrics(metrics : dict, path : str) -> None:
    try:
        with open(path, 'w') as f:
            json.dump(metrics, f)
    except Exception as e:  
        raise Exception(f"Error saving metrics: {e}")

def main():
    try:
        param_path = 'params.yaml'
        params = load_params(param_path)

        label = params['data']['label_column']

        test_data_path = params['data']['test_data_path']
        model_path = params['model_path']
        metrics_path = params['metrics_path']

        test_data = load_data(test_data_path)

        X_test, y_test = split_label(test_data, label)

        model = load_model(model_path)
        y_pred = predict(model, X_test)

        acc, pres, recall, f1 = evaluate(y_test, y_pred)
        metrics_dict = to_dict(acc, pres, recall, f1)

        save_metrics(metrics_dict,metrics_path)
    except Exception as e:
        raise Exception(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()