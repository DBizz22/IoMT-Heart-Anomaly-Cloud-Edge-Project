from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import ray
from ray.data import Dataset
import yaml

from LR_model import run_experiment as run_lr_experiment
from RF_model import run_experiment as run_rf_experiment
from XGBoost_model import run_experiment as run_xgb_experiment
from observability import init_tracing


DEFAULT_CONFIG_PATH = "config.yaml"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"config file not found : {config_path_obj}")

    with open(config_path_obj, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_dataframe_with_ray(cfg: Dict[str, Any]) -> pd.DataFrame:
    data_cfg = cfg.get("data", {})
    dataset_path = data_cfg.get("dataset_path", "dataset/heart_failure_dataset.csv")
    ds: Dataset = ray.data.read_csv(dataset_path)
    frame = ds.to_pandas()

    label_column = data_cfg.get("label_column", "HeartDisease")
    if label_column not in frame.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset columns.")

    return frame


@ray.remote
def _run_lr_remote(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return run_lr_experiment(df=df, cfg=cfg)


@ray.remote
def _run_rf_remote(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return run_rf_experiment(df=df, cfg=cfg)


@ray.remote
def _run_xgb_remote(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return run_xgb_experiment(df=df, cfg=cfg)


def run_distributed_training(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    tracer = init_tracing("iomt-distributed-training")

    if tracer:
        with tracer.start_as_current_span("load_config"):
            cfg = load_config(config_path)
    else:
        cfg = load_config(config_path)

    ray_cfg = cfg.get("ray", {})
    if not ray.is_initialized():
        log_level_name = str(ray_cfg.get("logging_level", "CRITICAL")).upper()
        log_level = getattr(logging, log_level_name, logging.CRITICAL)
        ray.init(
            ignore_reinit_error=ray_cfg.get("ignore_reinit_error", True),
            log_to_driver=ray_cfg.get("log_to_driver", False),
            configure_logging=ray_cfg.get("configure_logging", True),
            logging_level=log_level,
        )

    if tracer:
        with tracer.start_as_current_span("load_dataset"):
            df = load_dataframe_with_ray(cfg)
    else:
        df = load_dataframe_with_ray(cfg)

    df_ref = ray.put(df)
    cfg_ref = ray.put(cfg)

    if tracer:
        with tracer.start_as_current_span("run_parallel_models"):
            futures = {
                "logistic_regression": _run_lr_remote.remote(df_ref, cfg_ref),
                "random_forest": _run_rf_remote.remote(df_ref, cfg_ref),
                "xgboost": _run_xgb_remote.remote(df_ref, cfg_ref),
            }
            results = {name: ray.get(ref) for name, ref in futures.items()}
    else:
        futures = {
            "logistic_regression": _run_lr_remote.remote(df_ref, cfg_ref),
            "random_forest": _run_rf_remote.remote(df_ref, cfg_ref),
            "xgboost": _run_xgb_remote.remote(df_ref, cfg_ref),
        }
        results = {name: ray.get(ref) for name, ref in futures.items()}

    summary = {
        "dataset": cfg.get("data", {}).get("dataset_path"),
        "label_column": cfg.get("data", {}).get("label_column", "HeartDisease"),
        "models": results,
    }

    output_path = Path("results") / "distributed_training_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["report_path"] = str(output_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run distributed hyperparameterized training for Logistic, RandomForest, and XGBoost models."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration yaml file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = run_distributed_training(config_path=args.config)

    print("Distributed training completed.")
    print(f"Report: {report['report_path']}")
    for model_name, model_result in report["models"].items():
        metrics = model_result.get("metrics", {})
        print(
            f"- {model_name}: "
            f"f1={metrics.get('f1_label_1', 0.0):.4f}, "
            f"recall={metrics.get('recall_label_1', 0.0):.4f}, "
            f"roc_auc={metrics.get('roc_auc', 0.0):.4f}"
        )
