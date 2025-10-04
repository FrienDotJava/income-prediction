import os
import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from income_classification.helper.data_helper import load_data, split_label
from income_classification.helper.param_helper import load_params
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
# import dagshub
# dagshub.init(repo_owner='FrienDotJava', repo_name='income-prediction', mlflow=True)

# mlflow.set_tracking_uri("https://dagshub.com/FrienDotJava/income-prediction.mlflow")
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("best_model_GB")

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

def save_confusion_matrix(cm, path: str) -> None:
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(path)
        mlflow.log_artifact(path)
    except Exception as e:
        raise Exception(f"Error saving confusion matrix: {e}")

def main():
    try:
        with mlflow.start_run() as run:
            param_path = 'params.yaml'
            params = load_params(param_path)

            label = params['data']['label_column']

            test_data_path = params['data']['test_data_path']
            model_path = params['model_path']
            metrics_path = params['metrics_path']

            test_size = params['data_preprocessing']['test_size']
            random_state = params['data_preprocessing']['random_state']

            n_estimators = params['model_training']['n_estimators']
            max_depth = params['model_training']['max_depth']
            learning_rate = params['model_training']['learning_rate']

            test_data = load_data(test_data_path)

            X_test, y_test = split_label(test_data, label)

            model = load_model(model_path)
            y_pred = predict(model, X_test)

            acc, pres, recall, f1 = evaluate(y_test, y_pred)
            metrics_dict = to_dict(acc, pres, recall, f1)

            os.makedirs('reports', exist_ok=True) 
            save_metrics(metrics_dict,metrics_path)

            mlflow.log_metrics(metrics_dict)
            mlflow.log_params({
                'test_size': test_size,
                'random_state': random_state,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate
            })

            cm = confusion_matrix(y_test, y_pred)
            save_confusion_matrix(cm, params['cm_path'])

            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            
            signature = infer_signature(X_test, y_pred)

            mlflow.sklearn.log_model(model, name="GradientBoostingClassifier_bestModel", signature=signature)
            # mlflow.log_artifact(model_path, artifact_path="model")

            run_info = {'run_id': run.info.run_id, 'model_name': "GradientBoostingClassifier_bestModel"}
            reports_path = "reports/run_info.json"
            with open(reports_path, 'w') as file:
                json.dump(run_info, file, indent=4)
            
    except Exception as e:
        raise Exception(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()