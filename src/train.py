import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

def train_evaluate_model(X_train, X_test, y_train, y_test, model_name, model_params):
    mlflow.log_param("model_name", model_name)

    if model_name == "LogisticRegression":
        model = LogisticRegression(**model_params)
    elif model_name == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(**model_params)
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier(**model_params)
    elif model_name == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(**model_params)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })

    mlflow.sklearn.log_model(model, "model")

    return {
        "model": model,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
    }

def run_training_experiment(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    models_to_train = {
        "LogisticRegression": {
            "params": {"solver": "liblinear", "random_state": random_state, "class_weight": "balanced"},
            "grid": {"C": [0.1, 1.0, 10.0]}
        },
        "DecisionTreeClassifier": {
            "params": {"random_state": random_state, "class_weight": "balanced"},
            "grid": {"max_depth": [5, 10, 15]}
        },
        "LogisticRegression_tuned": {
            "params": {"solver": "liblinear", "random_state": random_state, "class_weight": "balanced"},
            "grid": {"C": [0.01, 0.1, 1.0, 10.0]}
        },
        "RandomForestClassifier_tuned": {
            "params": {"random_state": random_state, "class_weight": "balanced"},
            "grid": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
        }
    }

    best_model_overall = None
    best_roc_auc = -1

    for model_name, config in models_to_train.items():
        with mlflow.start_run(run_name=f"{model_name}_tuning"):
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)

            if "tuned" in model_name:
                base_model_class = None
                if "LogisticRegression" in model_name:
                    base_model_class = LogisticRegression
                elif "RandomForestClassifier" in model_name:
                    base_model_class = RandomForestClassifier

                if base_model_class:
                    grid_search = GridSearchCV(base_model_class(**config["params"]), config["grid"], cv=3, scoring='roc_auc', n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    best_params = grid_search.best_params_
                    mlflow.log_params(best_params)
                    print(f"Best params for {model_name}: {best_params}")

                    result = train_evaluate_model(X_train, X_test, y_train, y_test, model_name.replace("_tuned", ""), best_params)
                else:
                    print(f"No base model class found for {model_name}")
                    continue
            else:
                result = train_evaluate_model(X_train, X_test, y_train, y_test, model_name, config["params"])

            current_roc_auc = result["metrics"]["roc_auc"]
            if current_roc_auc > best_roc_auc:
                best_roc_auc = current_roc_auc
                best_model_overall = result["model"]

    if best_model_overall:
        with mlflow.start_run(run_name="Best_Model_Final_Run"):
            mlflow.log_param("final_model_name", best_model_overall.__class__.__name__)
            mlflow.log_metric("final_roc_auc", best_roc_auc)

            mlflow.sklearn.log_model(
                best_model_overall,
                "best_credit_risk_model",
                registered_model_name="CreditRiskProxyModel"
            )
            print(f"Best model ({best_model_overall.__class__.__name__}) registered with ROC-AUC: {best_roc_auc}")
    return best_model_overall