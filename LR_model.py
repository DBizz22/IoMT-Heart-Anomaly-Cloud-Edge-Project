from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def _build_preprocessor(
    df: pd.DataFrame, label_column: str, cfg: Dict[str, Any]
) -> ColumnTransformer:
    encoding_cfg = cfg.get("encoding", {})
    ordinal_mappings = encoding_cfg.get("ordinal_mappings", {})
    drop_first = "first" if encoding_cfg.get("drop_first_baseline", True) else None

    X = df.drop(columns=[label_column])
    ordinal_columns = [c for c in ordinal_mappings.keys() if c in X.columns]
    object_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    nominal_columns = [c for c in object_columns if c not in ordinal_columns]
    numeric_columns = [
        c for c in X.columns if c not in ordinal_columns and c not in nominal_columns
    ]

    transformers = []
    if nominal_columns:
        transformers.append(
            (
                "nominal",
                OneHotEncoder(drop=drop_first, handle_unknown="ignore"),
                nominal_columns,
            )
        )

    if ordinal_columns:
        ordinal_orders = [ordinal_mappings[col] for col in ordinal_columns]
        transformers.append(
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_orders,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_columns,
            )
        )

    if numeric_columns and cfg.get("preprocessing", {}).get("scale_enabled", True):
        transformers.append(("numeric", StandardScaler(), numeric_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def run_experiment(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {})
    label_column = data_cfg.get("label_column", "HeartDisease")
    split_seed = int(data_cfg.get("split_seed", 42))
    test_size = float(data_cfg.get("test_size", 0.2))

    X = df.drop(columns=[label_column])
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=split_seed,
        stratify=y,
    )

    preprocessor = _build_preprocessor(df, label_column, cfg)

    scorer = make_scorer(fbeta_score, beta=1.5, pos_label=1, average="binary")
    base_model = LogisticRegression(
        solver="lbfgs",
        max_iter=1500,
        random_state=split_seed,
        class_weight={0: 1.0, 1: 1.4},
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("logisticregression", base_model),
        ],
        memory=None,
    )

    param_grid = {
        "logisticregression__C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "logisticregression__class_weight": [
            {0: 1.0, 1: 1.2},
            {0: 1.0, 1: 1.4},
            {0: 1.0, 1: 1.6},
        ],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, _, _ = cm.ravel()

    output_path = Path("results") / "logistic_regression" / "model.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_path)

    return {
        "model": "logistic_regression",
        "best_params": search.best_params_,
        "best_cv_score_fbeta15": float(search.best_score_),
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_label_1": float(
                precision_score(y_test, y_pred, zero_division=0)
            ),
            "recall_label_1": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_label_1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "confusion_matrix": cm.tolist(),
        },
        "artifact_path": str(output_path),
    }
