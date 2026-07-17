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
from WorldCupPredictor.src.models.train import split_train_test, train


PENALTIES = ["l1", "l2"]
C_VALUES = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
PENALTY_L1_RATIOS = {
    "l1": 1.0,
    "l2": 0.0,
}


def build_model(penalty: str, c_value: float) -> Pipeline:
    """Create a scaled logistic regression model for one tuning config."""
    if penalty not in PENALTY_L1_RATIOS:
        raise ValueError(f"Unsupported penalty: {penalty}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    C=c_value,
                    l1_ratio=PENALTY_L1_RATIOS[penalty],
                    solver="saga",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def score_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return the core metrics for a trained classifier."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return {
        "accuracy": model.score(X_test, y_test),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "macro_roc_auc": roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr",
            average="macro",
        ),
    }


def tune_logistic_regression(
    penalties: list[str] = PENALTIES,
    c_values: list[float] = C_VALUES,
) -> pd.DataFrame:
    """Train and evaluate logistic regression across penalties and C values."""
    feature_table = build_basic_feature_table()
    X, y = split_features_and_target(feature_table)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    results = []
    for penalty in penalties:
        for c_value in c_values:
            print(f"Training penalty={penalty}, C={c_value}...")
            model = build_model(penalty=penalty, c_value=c_value)
            model = train(X_train, y_train, model=model)
            scores = score_model(model, X_test, y_test)

            result = {
                "penalty": penalty,
                "C": c_value,
                **scores,
            }
            results.append(result)

            print(
                "  "
                f"accuracy={scores['accuracy']:.4f}, "
                f"macro_f1={scores['macro_f1']:.4f}, "
                f"macro_roc_auc={scores['macro_roc_auc']:.4f}"
            )

    return pd.DataFrame(results).sort_values(
        by=["accuracy", "macro_f1", "macro_roc_auc"],
        ascending=False,
    )


if __name__ == "__main__":
    tuning_results = tune_logistic_regression()

    print("\nBest configs:")
    print(tuning_results.to_string(index=False))
