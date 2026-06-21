from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from preprocessing import (
    PROFIT_MARGIN_THRESHOLD,
    RAW_FEATURES,
    TARGET_COLUMN,
    OrderRiskPreprocessor,
    clean_training_data,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
ARTIFACT_DIR = APP_DIR / "artifacts"
RAW_DATA_PATH = PROJECT_DIR / "Sales Dataset.csv"
RANDOM_STATE = 42
PREDICTION_THRESHOLD = 0.5


CATBOOST_PARAMS = {
    "iterations": 400,
    "learning_rate": 0.05367348098271405,
    "depth": 6,
    "l2_leaf_reg": 0.030385687558382078,
    "random_strength": 0.04614512934665785,
    "bagging_temperature": 0.33433608718412866,
    "border_count": 254,
    "loss_function": "Logloss",
    "eval_metric": "F1",
    "random_seed": RANDOM_STATE,
    "verbose": False,
    "allow_writing_files": False,
}


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(RAW_DATA_PATH)
    clean_df = clean_training_data(raw_df)

    X = clean_df[RAW_FEATURES]
    y = clean_df[TARGET_COLUMN].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = OrderRiskPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= PREDICTION_THRESHOLD).astype(int)

    metrics = {
        "model": "CatBoostClassifier",
        "target_rule": f"{TARGET_COLUMN}=1 if Profit_Margin >= {PROFIT_MARGIN_THRESHOLD}, else 0",
        "prediction_threshold": PREDICTION_THRESHOLD,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "feature_columns": X_train.columns.tolist(),
        "catboost_params": CATBOOST_PARAMS,
    }

    model.save_model(ARTIFACT_DIR / "catboost_order_risk.cbm")
    joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.joblib")
    (ARTIFACT_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Saved model artifacts to:", ARTIFACT_DIR)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
