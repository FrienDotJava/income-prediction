import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from income_classification.helper.data_helper import load_data, split_label
from income_classification.helper.param_helper import load_params
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import pandas as pd
import pickle

from income_classification.modeling.model_evaluation import load_model

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

def hyperparameter_tuning(model: GradientBoostingClassifier, X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    try:
        param_dist = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }

        random_search = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), param_distributions=param_dist, cv=5, n_iter=27, scoring='accuracy', n_jobs=-1, verbose=2)
        random_search.fit(X_train, y_train)

        # best_params = random_search.best_params_
        # best_gb = random_search.best_estimator_
        return random_search
    except Exception as e:
        raise Exception(f"Error during hyperparameter tuning: {e}")

def save_model(model : GradientBoostingClassifier, save_path: str) -> None:
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        raise Exception(f"Error saving model: {e}")

def create_pipeline(train_set_dropped: pd.DataFrame, model_path: str) -> Pipeline:
    try:
        X = train_set_dropped.drop('income', axis=1)
        y = train_set_dropped['income']

        num_cols = X.select_dtypes('number').columns.tolist()
        cat_cols = X.select_dtypes('object').columns.tolist()


        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", MinMaxScaler(), num_cols),
            ],
            remainder="drop"
        )

        gbc = load_model(model_path)

        pipe = Pipeline([("pre", preprocessor), ("clf", gbc)])

        pipe.fit(X, y)

        joblib.dump(pipe, "pipeline_model.pkl")
        print("Saved pipeline_model.pkl")
    except Exception as e:
        raise Exception(f"Error creating pipeline: {e}")
    
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
        dropped_columns_path_train = params['data']['dropped_columns_path_train']

        train_set_dropped = load_data(dropped_columns_path_train)
        train_data = load_data(train_data_path)

        X_train, y_train = split_label(train_data, label)

        model = initiate_model(n_estimators, max_depth, learning_rate)

        trained_model = train_model(model, X_train, y_train)

        os.makedirs('models', exist_ok=True) 
        save_model(trained_model, model_path)

        create_pipeline(train_set_dropped, model_path)
    except Exception as e:
        raise Exception(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
    




