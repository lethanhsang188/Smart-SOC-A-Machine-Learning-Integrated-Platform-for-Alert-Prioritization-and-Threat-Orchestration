"""FastAPI service for Wazuh alert priority inference."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REQUIRED_IMPORTS = {
    "joblib": "joblib>=1.3.0",
    "sklearn": "scikit-learn>=1.3.0",
    "fastapi": "fastapi>=0.110.0",
    "pydantic": "pydantic>=2.0.0",
    "uvicorn": "uvicorn[standard]>=0.27.0",
}


def ensure_dependencies() -> None:
    missing = []
    for module_name, package_spec in REQUIRED_IMPORTS.items():
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(package_spec)
    if not missing:
        return
    print(f"Installing missing Python packages: {', '.join(missing)}", file=sys.stderr)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_dependencies()

import joblib  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from feature_extractor import extract_features  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "alert_priority_model.joblib"
MODEL_PATH = Path(os.getenv("ALERT_PRIORITY_MODEL_PATH", DEFAULT_MODEL_PATH)).resolve()
NOTIFY_THRESHOLD = float(os.getenv("ALERT_PRIORITY_NOTIFY_THRESHOLD", "0.70"))


class AlertRequest(BaseModel):
    alert: Dict[str, Any]


class BatchAlertRequest(BaseModel):
    alerts: List[Dict[str, Any]] = Field(default_factory=list)


class PriorityPrediction(BaseModel):
    priority: str
    score: float
    confidence: float
    class_probabilities: Dict[str, float]
    should_notify: bool
    block_ip: Optional[str] = None
    block_ip_source: Optional[str] = None
    model_version: str
    timestamp: str


class ModelManager:
    _bundle: Optional[Dict[str, Any]] = None

    @classmethod
    def bundle(cls) -> Dict[str, Any]:
        if cls._bundle is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model not found: {MODEL_PATH}. Train or copy the model to {DEFAULT_MODEL_PATH}, "
                    "or set ALERT_PRIORITY_MODEL_PATH to the .joblib file."
                )
            cls._bundle = joblib.load(MODEL_PATH)
        return cls._bundle


def priority_score(priority: str, confidence: float) -> float:
    base = {"P1": 0.95, "P2": 0.82, "P3": 0.55, "P4": 0.20}.get(priority, 0.20)
    return round(min(1.0, max(0.0, (base * 0.75) + (confidence * 0.25))), 4)


def get_nested(data: Dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def extract_block_ip(alert: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Pick the attacker/source IP across web accesslog and Suricata alerts."""
    candidates = [
        ("data.srcip", get_nested(alert, "data.srcip")),
        ("data.src_ip", get_nested(alert, "data.src_ip")),
        ("srcip", alert.get("srcip")),
        ("src_ip", alert.get("src_ip")),
        ("data.flow.src_ip", get_nested(alert, "data.flow.src_ip")),
        ("data.srcport_host", get_nested(alert, "data.srcport_host")),
    ]
    for source, value in candidates:
        if value is None:
            continue
        ip = str(value).strip()
        if ip:
            return ip, source
    return None, None


def predict_one(alert: Dict[str, Any]) -> Dict[str, Any]:
    bundle = ModelManager.bundle()
    pipeline = bundle["pipeline"]
    metadata = bundle.get("metadata", {})
    features = extract_features(alert)
    priority = str(pipeline.predict([features])[0])

    probabilities: Dict[str, float] = {}
    confidence = 0.0
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([features])[0]
        classes = list(pipeline.classes_)
        probabilities = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}
        confidence = float(max(proba))
    score = priority_score(priority, confidence)
    should_notify = priority in {"P1", "P2"} or score >= NOTIFY_THRESHOLD
    block_ip, block_ip_source = extract_block_ip(alert)

    return {
        "priority": priority,
        "score": score,
        "confidence": round(confidence, 4),
        "class_probabilities": probabilities,
        "should_notify": should_notify,
        "block_ip": block_ip,
        "block_ip_source": block_ip_source,
        "model_version": metadata.get("version", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


app = FastAPI(title="Wazuh Alert Priority ML Service", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "python": sys.executable,
    }


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    try:
        bundle = ModelManager.bundle()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return bundle.get("metadata", {})


@app.post("/predict", response_model=PriorityPrediction)
def predict(req: AlertRequest) -> Dict[str, Any]:
    try:
        return predict_one(req.alert)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict-batch")
def predict_batch(req: BatchAlertRequest) -> Dict[str, Any]:
    try:
        return {"results": [predict_one(alert) for alert in req.alerts]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("ALERT_PRIORITY_HOST", "0.0.0.0")
    port = int(os.getenv("ALERT_PRIORITY_PORT", "8010"))
    uvicorn.run("alert_priority_server:app", host=host, port=port)
