from __future__ import annotations

import argparse
import os
import time
from threading import Lock
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ray import serve

from observability import init_tracing


app = FastAPI(title="IoMT MLflow Prediction API", version="1.0.0")
tracer = init_tracing("iomt-inference-api")


class PredictRequest(BaseModel):
    instances: list[dict[str, Any]] = Field(
        ...,
        description="List of feature dictionaries to score. Keys must match model input columns.",
        min_length=1,
    )


class PredictResponse(BaseModel):
    model_uri: str
    predictions: list[Any]
    count: int


def _to_jsonable(values: Any) -> list[Any]:
    """Convert NumPy/Pandas outputs into JSON-serializable Python values."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    converted: list[Any] = []
    for v in arr.tolist():
        if isinstance(v, (np.floating, float)):
            converted.append(float(v))
        elif isinstance(v, (np.integer, int)):
            converted.append(int(v))
        elif isinstance(v, (np.bool_, bool)):
            converted.append(bool(v))
        else:
            converted.append(v)
    return converted


def _read_dotenv_value(key: str, dotenv_path: str = ".env") -> str | None:
    path = Path(dotenv_path)
    if not path.exists():
        return None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


def resolve_tracking_uri(cli_tracking_uri: str | None) -> str:
    if cli_tracking_uri:
        return cli_tracking_uri

    # In local runs, prefer project .env over inherited shell variables.
    in_kubernetes = bool(os.getenv("KUBERNETES_SERVICE_HOST"))
    if not in_kubernetes:
        dotenv_value = _read_dotenv_value("MLFLOW_TRACKING_URI")
        if dotenv_value:
            return dotenv_value

    return os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def resolve_bool_setting(
    key: str,
    default: bool = False,
    dotenv_path: str = ".env",
) -> bool:
    # In local runs, prefer project .env over inherited shell variables.
    in_kubernetes = bool(os.getenv("KUBERNETES_SERVICE_HOST"))
    if not in_kubernetes:
        dotenv_value = _read_dotenv_value(key, dotenv_path=dotenv_path)
        if dotenv_value is not None:
            return dotenv_value.strip().lower() in {"1", "true", "yes", "on"}

    env_value = os.getenv(key)
    if env_value is None:
        return default
    return env_value.strip().lower() in {"1", "true", "yes", "on"}


@serve.deployment(name="mlflow_predictor", num_replicas=1)
@serve.ingress(app)
class MLflowPredictor:
    def __init__(self, model_uri: str, tracking_uri: str | None = None) -> None:
        self.model_uri = model_uri
        self.tracking_uri = tracking_uri
        self.model_backend = "unknown"
        self.model = None
        self._model_lock = Lock()
        self._model_load_error: str | None = None

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        preload = resolve_bool_setting("PREDICTOR_PRELOAD_MODEL", default=False)
        if preload:
            self.model = self._load_model(model_uri)

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return

        with self._model_lock:
            if self.model is not None:
                return
            try:
                self.model = self._load_model(self.model_uri)
                self._model_load_error = None
            except Exception as exc:
                self._model_load_error = str(exc)
                raise

    def _load_model(self, model_uri: str):
        # Primary: MLflow pyfunc
        model = mlflow.pyfunc.load_model(model_uri)
        if model is not None and hasattr(model, "predict"):
            self.model_backend = "mlflow.pyfunc"
            return model

        # Secondary: sklearn flavor from MLflow
        try:
            sk_model = mlflow_sklearn.load_model(model_uri)
            if sk_model is not None and hasattr(sk_model, "predict"):
                self.model_backend = "mlflow.sklearn"
                return sk_model
        except Exception:
            pass

        # Fallback: local joblib artifact path if provided directly
        path = Path(model_uri)
        if path.exists() and path.suffix in {".joblib", ".pkl"}:
            import joblib

            jb_model = joblib.load(path)
            if jb_model is not None and hasattr(jb_model, "predict"):
                self.model_backend = "joblib"
                return jb_model

        raise ValueError(
            f"Unable to load a valid prediction model from '{model_uri}'. "
            "Loaded object was None or missing 'predict()'."
        )

    @app.get("/health")
    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_uri": self.model_uri,
            "model_backend": self.model_backend,
            "model_loaded": self.model is not None,
            "model_load_error": self._model_load_error,
            "tracking_uri": mlflow.get_tracking_uri(),
        }

    @app.post(
        "/predict",
        responses={
            400: {"description": "Invalid payload or prediction failure."},
            500: {"description": "Model is not loaded correctly."},
        },
    )
    def predict(self, payload: PredictRequest) -> PredictResponse:
        if not payload.instances:
            raise HTTPException(status_code=400, detail="'instances' cannot be empty.")

        if self.model is None:
            try:
                self._ensure_model_loaded()
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model failed to load from '{self.model_uri}': {exc}",
                ) from exc

        if self.model is None or not hasattr(self.model, "predict"):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Model is not loaded correctly (None or missing predict). "
                    f"model_uri={self.model_uri} backend={self.model_backend}"
                ),
            )

        try:
            if tracer:
                with tracer.start_as_current_span("predict_request") as span:
                    span.set_attribute("iomt.request.instances", len(payload.instances))
                    frame = pd.DataFrame(payload.instances)
                    predictions = self.model.predict(frame)
            else:
                frame = pd.DataFrame(payload.instances)
                predictions = self.model.predict(frame)

            values = _to_jsonable(predictions)
            return PredictResponse(
                model_uri=self.model_uri,
                predictions=values,
                count=len(values),
            )
        except Exception as exc:  # pragma: no cover
            raise HTTPException(
                status_code=400, detail=f"Prediction failed: {exc}"
            ) from exc


def build_deployment(model_uri: str, tracking_uri: str | None = None):
    return MLflowPredictor.bind(model_uri=model_uri, tracking_uri=tracking_uri)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy an MLflow-registered model using Ray Serve + FastAPI."
    )
    parser.add_argument(
        "--model-uri",
        default=os.getenv(
            "MODEL_URI", "models:/HeartDiseaseRandomForestTuned@champion"
        ),
        help="MLflow model URI, e.g. models:/MyModel@Production or models:/MyModel/1",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI (file://... or http://...).",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("SERVE_HOST", "127.0.0.1"),
        help="Ray Serve HTTP host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("SERVE_PORT", "8000")),
        help="Ray Serve HTTP port.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tracking_uri = resolve_tracking_uri(args.tracking_uri)

    # Local single-node startup (switch to address='auto' for existing clusters).
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    mlflow.set_tracking_uri(tracking_uri)

    serve.start(http_options={"host": args.host, "port": args.port})
    serve.run(
        build_deployment(model_uri=args.model_uri, tracking_uri=tracking_uri),
        name="mlflow_prediction_api",
        route_prefix="/",
    )

    print("Ray Serve app is running.")
    print(f"Model URI: {args.model_uri}")
    print(f"Tracking URI: {args.tracking_uri}")
    print(f"Health endpoint: http://{args.host}:{args.port}/health")
    print(f"Predict endpoint: http://{args.host}:{args.port}/predict")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
