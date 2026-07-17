import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from WorldCupPredictor.src.features.feature_builder import (
    build_basic_feature_table,
    split_features_and_target,
)

def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the features and target into training and testing sets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

def train(X_train: pd.DataFrame, y_train: pd.Series, model):
    """Train a logistic regression model on the training data."""

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate the trained model on the test data."""
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) 


    accuracy = model.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(
    y_test, 
    y_prob, 
    multi_class='ovr',  # "One-vs-Rest" strategies compare each class against all others
    average='macro'     # Computes the score for each class independently and averages them
    )


    print(f"Multi-class ROC-AUC Score: {roc_auc:.4f}")

    print(f"Model accuracy on test set: {accuracy:.4f}")
    print(f"Model F1 Score: {f1:.4f}")




if __name__ == "__main__":

    feature_table = build_basic_feature_table()
    X, y = split_features_and_target(feature_table)
    
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=5000,
        C=5.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42)), 
    ])
    model = train(X_train, y_train, model=model)
    evaluate_model(model, X_test, y_test)
