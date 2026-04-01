from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, cast

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
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
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


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
    xgb_cfg = cfg.get("training", {}).get("xgboost", {})

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
    scorer = make_scorer(
        fbeta_score,
        beta=float(xgb_cfg.get("fbeta_beta", 1.2)),
        pos_label=1,
        average="binary",
    )

    xgb_model = XGBClassifier(
        objective=xgb_cfg.get("objective", "binary:logistic"),
        eval_metric=xgb_cfg.get("eval_metric", ["logloss", "error"]),
        tree_method="hist",
        random_state=split_seed,
        n_jobs=1,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("xgb", xgb_model),
        ],
        memory=None,
    )

    max_depth_low, max_depth_high = xgb_cfg.get("max_depth_range", [3, 8])
    subsample_low, subsample_high = xgb_cfg.get("subsample_range", [0.7, 1.0])
    eta_low, eta_high = xgb_cfg.get("eta_log_range", [0.003, 0.08])

    param_distributions = {
        "xgb__max_depth": list(range(int(max_depth_low), int(max_depth_high) + 1)),
        "xgb__min_child_weight": xgb_cfg.get("min_child_weight_values", [2, 4, 6, 8]),
        "xgb__subsample": [round(v, 3) for v in [subsample_low, 0.8, subsample_high]],
        "xgb__colsample_bytree": xgb_cfg.get(
            "colsample_bytree_values", [0.7, 0.85, 1.0]
        ),
        "xgb__learning_rate": [round(v, 5) for v in [eta_low, 0.01, 0.03, eta_high]],
        "xgb__gamma": xgb_cfg.get("gamma_values", [0.0, 0.5, 1.0]),
        "xgb__reg_lambda": xgb_cfg.get("reg_lambda_values", [1.0, 2.0, 5.0, 10.0]),
        "xgb__reg_alpha": xgb_cfg.get("reg_alpha_values", [0.0, 0.1, 0.5]),
        "xgb__scale_pos_weight": xgb_cfg.get(
            "scale_pos_weight_values", [1.0, 1.2, 1.4, 1.6]
        ),
        "xgb__n_estimators": xgb_cfg.get("num_boost_round_values", [200, 400, 600]),
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=int(xgb_cfg.get("num_samples", 12)),
        scoring=scorer,
        cv=5,
        random_state=split_seed,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_model = cast(Pipeline, search.best_estimator_)
    threshold = float(xgb_cfg.get("decision_threshold", 0.5))
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, _, _ = cm.ravel()

    model_dir = Path("results") / "xgboost"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(best_model, model_path)

    return {
        "model": "xgboost",
        "best_params": search.best_params_,
        "best_cv_score_fbeta": float(search.best_score_),
        "decision_threshold": threshold,
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
        "artifact_path": str(model_path),
    }
