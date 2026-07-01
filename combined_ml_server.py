"""
FastAPI server gop 2 chuc nang:
  1. Phat hien malware PE theo EMBER2024 tu ml_server.py
  2. Du doan muc uu tien alert Wazuh tu alert_priority_server.py

Thuat toan du doan duoc giu nguyen. File nay chi tach ten nhung phan bi trung
giua 2 file goc, vi du ModelManager, health va model-info.


Endpoint chinh:
    POST /predict-features       Du doan malware tu feature PE
    POST /predict                Du doan muc uu tien alert
    POST /predict-batch          Du doan muc uu tien cho nhieu alert
    GET  /health                 Kiem tra suc khoe toan bo service
    GET  /model-info             Thong tin ca 2 model
    GET  /malware/model-info     Thong tin model malware
    GET  /alert-priority/model-info
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REQUIRED_IMPORTS = {
    "joblib": "joblib>=1.3.0",
    "sklearn": "scikit-learn>=1.3.0",
    "lightgbm": "lightgbm",
    "numpy": "numpy",
    "pandas": "pandas",
    "requests": "requests",
    "urllib3": "urllib3",
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import urllib3  # noqa: E402
from fastapi import Body, FastAPI, Header, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

try:
    import lightgbm as lgb  # noqa: E402
except ImportError:
    raise RuntimeError("pip install lightgbm")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
ALERT_PRIORITY_DIR = BASE_DIR / "alert_priority"
if str(ALERT_PRIORITY_DIR) not in sys.path:
    sys.path.insert(0, str(ALERT_PRIORITY_DIR))

from feature_extractor import derive_priority_label, extract_features, normalize_attack_type  # noqa: E402


# =============================================================================
# CAU HINH PHAT HIEN MALWARE
# =============================================================================

DEFAULT_MALWARE_MODEL_PATH = BASE_DIR / "ember2024_pe_lgbm.model"
DEFAULT_MALWARE_LOG_CSV = BASE_DIR / "logs" / "malware_logs.csv"

MODEL_CONFIG = {
    "model_path": os.getenv("ML_MODEL_PATH", str(DEFAULT_MALWARE_MODEL_PATH)),
    "num_trees": 999,
    "num_features": 2560,
    "best_iteration": -1,
    "auc_roc": 0.9983184980315499,
    "accuracy": 0.9806407407407407,
    "fpr": 0.012303703703703704,
    "fnr": 0.026414814814814815,
    "f1_malware": 0.9805031798429661,
    "f1_benign": 0.9807763741012486,
    "precision_malware": 0.9875201923438146,
    "recall_malware": 0.9735851851851852,
    "confusion_matrix": {
        "TN": 533356,
        "FP": 6644,
        "FN": 14264,
        "TP": 525736,
    },
    "categorical_feature_indices": [2, 3, 4, 5, 6, 701, 702],
    "top_features": [
        {"rank": 1, "feature": "Column_739", "gain": 12856688.58},
        {"rank": 2, "feature": "Column_2413", "gain": 6572892.55},
        {"rank": 3, "feature": "Column_2539", "gain": 1972618.21},
        {"rank": 4, "feature": "Column_2414", "gain": 1829650.54},
        {"rank": 5, "feature": "Column_993", "gain": 956657.88},
        {"rank": 6, "feature": "Column_509", "gain": 831880.35},
        {"rank": 7, "feature": "Column_994", "gain": 781334.71},
        {"rank": 8, "feature": "Column_508", "gain": 738421.92},
        {"rank": 9, "feature": "Column_2405", "gain": 638726.59},
        {"rank": 10, "feature": "Column_991", "gain": 615930.90},
    ],
}

MALWARE_THRESHOLD = float(os.getenv("ML_MALWARE_THRESHOLD", "0.5"))
SUSPICIOUS_TOOL_THRESHOLD = float(os.getenv("ML_SUSPICIOUS_TOOL_THRESHOLD", "0.30"))
HIGH_RISK_TOOL_NAMES = {
    "mimikatz.exe",
    "mimidrv.sys",
}
HIGH_RISK_PATH_TOKENS = {
    "mimikatz-master",
}

SHUFFLE_MALWARE_WEBHOOK = os.getenv(
    "SHUFFLE_MALWARE_WEBHOOK",
    "",
).strip()
SHUFFLE_VERIFY_SSL = os.getenv("SHUFFLE_VERIFY_SSL", "0").lower() in ("1", "true", "yes")
SHUFFLE_CA_BUNDLE = os.getenv("SHUFFLE_CA_BUNDLE", "").strip()
LOG_CSV = os.getenv("ML_LOG_CSV", str(DEFAULT_MALWARE_LOG_CSV))


# =============================================================================
# CAU HINH DU DOAN MUC UU TIEN ALERT
# =============================================================================

DEFAULT_ALERT_MODEL_PATH = ALERT_PRIORITY_DIR / "models" / "alert_priority_model.joblib"
ALERT_PRIORITY_MODEL_PATH = Path(
    os.getenv("ALERT_PRIORITY_MODEL_PATH", DEFAULT_ALERT_MODEL_PATH)
).resolve()
NOTIFY_THRESHOLD = float(os.getenv("ALERT_PRIORITY_NOTIFY_THRESHOLD", "0.70"))

INGEST_API_KEY = os.getenv("ML_INGEST_API_KEY", "").strip()
INGEST_BLOCK_PRIORITIES = {
    item.strip().upper()
    for item in os.getenv("ML_INGEST_BLOCK_PRIORITIES", "P1,P2").split(",")
    if item.strip()
}
INGEST_NOTIFY_PRIORITIES = {
    item.strip().upper()
    for item in os.getenv("ML_INGEST_NOTIFY_PRIORITIES", "P1,P2").split(",")
    if item.strip()
}
SHUFFLE_RESPONSE_WEBHOOK = os.getenv(
    "SHUFFLE_RESPONSE_WEBHOOK",
    "",
).strip()
SHUFFLE_RESPONSE_VERIFY_SSL = os.getenv("SHUFFLE_RESPONSE_VERIFY_SSL", "0").lower() in ("1", "true", "yes")
SHUFFLE_RESPONSE_TIMEOUT = float(os.getenv("SHUFFLE_RESPONSE_TIMEOUT", "8"))

PFSENSE_ALIAS_URL = os.getenv(
    "PFSENSE_ALIAS_URL",
    "",
).strip()
PFSENSE_API_KEY = os.getenv("PFSENSE_API_KEY", "").strip()
PFSENSE_ALIAS_NAME = os.getenv("PFSENSE_ALIAS_NAME", "WAZUH_BLOCK").strip()
PFSENSE_VERIFY_SSL = os.getenv("PFSENSE_VERIFY_SSL", "0").lower() in ("1", "true", "yes")
PFSENSE_TIMEOUT = float(os.getenv("PFSENSE_TIMEOUT", "8"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_TIMEOUT = float(os.getenv("TELEGRAM_TIMEOUT", "8"))


# =============================================================================
# KHAI BAO KIEU DU LIEU API
# =============================================================================

class PEInput(BaseModel):
    features: List[float]
    path: str
    agent_name: str = ""
    agent_ip: str = ""


class PredictionResult(BaseModel):
    path: str
    label: str
    probability: float
    threshold: float
    detection_reason: str = ""
    rule_hits: List[str] = Field(default_factory=list)
    agent_name: str
    agent_ip: str
    timestamp: str


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


# =============================================================================
# TAI MODEL MALWARE VA CHAY DU DOAN
# =============================================================================

class MalwareModelManager:
    _model = None

    @classmethod
    def get(cls) -> lgb.Booster:
        if cls._model is None:
            mp = os.path.abspath(os.path.expandvars(os.path.expanduser(MODEL_CONFIG["model_path"])))
            if not os.path.isfile(mp):
                raise FileNotFoundError(
                    f"Model khong tim thay tai: {mp}\n"
                    f"Hay dat file 'ember2024_pe_lgbm.model' vao thu muc: {BASE_DIR}\n"
                    f"Hoac set bien moi truong ML_MODEL_PATH den file model tren Windows."
                )
            logger.info(f"[+] Loading malware model: {mp}")
            cls._model = lgb.Booster(model_file=mp)
            logger.info(
                f"[+] Malware model loaded - trees={MODEL_CONFIG['num_trees']} "
                f"features={MODEL_CONFIG['num_features']} "
                f"AUC={MODEL_CONFIG['auc_roc']:.4f}"
            )
        return cls._model


def apply_detection_policy(path: str, probability: float) -> tuple[str, float, str, List[str]]:
    rule_hits: List[str] = []
    normalized_path = path.replace("/", "\\").lower()
    file_name = os.path.basename(normalized_path)

    if probability >= MALWARE_THRESHOLD:
        return "malicious", MALWARE_THRESHOLD, "ml_threshold", rule_hits

    if file_name in HIGH_RISK_TOOL_NAMES:
        rule_hits.append(f"high_risk_filename:{file_name}")

    for token in HIGH_RISK_PATH_TOKENS:
        if token in normalized_path:
            rule_hits.append(f"high_risk_path:{token}")

    if rule_hits and probability >= SUSPICIOUS_TOOL_THRESHOLD:
        return "malicious", SUSPICIOUS_TOOL_THRESHOLD, "high_risk_tool_policy", rule_hits

    return "legitimate", MALWARE_THRESHOLD, "ml_threshold", rule_hits


def run_inference(features: List[float], path: str = "") -> tuple[str, float, float, str, List[str]]:
    model = MalwareModelManager.get()

    n = MODEL_CONFIG["num_features"]
    if len(features) != n:
        raise ValueError(
            f"Feature dimension mismatch: nhan {len(features)}, "
            f"model can {n}"
        )

    x = np.array(features, dtype=np.float32).reshape(1, -1)
    prob = float(model.predict(x)[0])
    label, threshold, reason, rule_hits = apply_detection_policy(path, prob)
    return label, prob, threshold, reason, rule_hits


def log_to_csv(result: PredictionResult) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
        row = {
            "timestamp": result.timestamp,
            "agent_name": result.agent_name,
            "agent_ip": result.agent_ip,
            "path": result.path,
            "label": result.label,
            "probability": result.probability,
            "threshold": result.threshold,
            "detection_reason": result.detection_reason,
            "rule_hits": ";".join(result.rule_hits),
        }
        df = pd.DataFrame([row])
        write_header = not os.path.isfile(LOG_CSV)
        df.to_csv(LOG_CSV, mode="a", index=False, header=write_header)
    except Exception as e:
        logger.warning(f"[!] CSV logging failed: {e}")


def notify_shuffle(result: PredictionResult) -> None:
    verify_ssl: bool | str = SHUFFLE_CA_BUNDLE or SHUFFLE_VERIFY_SSL
    if verify_ssl is False:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    payload = {
        "agent_name": result.agent_name,
        "agent_ip": result.agent_ip,
        "path": result.path.replace("\\", "\\\\"),
        "label": result.label,
        "probability": result.probability,
        "threshold": result.threshold,
        "detection_reason": result.detection_reason,
        "rule_hits": result.rule_hits,
        "timestamp": result.timestamp,
        "model_info": {
            "num_trees": MODEL_CONFIG["num_trees"],
            "auc_roc": MODEL_CONFIG["auc_roc"],
            "fpr": MODEL_CONFIG["fpr"],
            "fnr": MODEL_CONFIG["fnr"],
            "threshold": MALWARE_THRESHOLD,
            "suspicious_tool_threshold": SUSPICIOUS_TOOL_THRESHOLD,
        },
    }
    try:
        resp = requests.post(
            SHUFFLE_MALWARE_WEBHOOK,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5,
            verify=verify_ssl,
        )
        resp.raise_for_status()
        logger.info(f"[+] Shuffle notified - status={resp.status_code}")
    except Exception as e:
        logger.warning(f"[!] Shuffle notification failed: {e}")


def malware_model_info_payload() -> Dict[str, Any]:
    return {
        "model_path": MODEL_CONFIG["model_path"],
        "num_trees": MODEL_CONFIG["num_trees"],
        "num_features": MODEL_CONFIG["num_features"],
        "threshold": MALWARE_THRESHOLD,
        "suspicious_tool_threshold": SUSPICIOUS_TOOL_THRESHOLD,
        "categorical_indices": MODEL_CONFIG["categorical_feature_indices"],
        "performance": {
            "auc_roc": MODEL_CONFIG["auc_roc"],
            "accuracy": MODEL_CONFIG["accuracy"],
            "fpr": MODEL_CONFIG["fpr"],
            "fnr": MODEL_CONFIG["fnr"],
            "f1_malware": MODEL_CONFIG["f1_malware"],
            "f1_benign": MODEL_CONFIG["f1_benign"],
            "precision_malware": MODEL_CONFIG["precision_malware"],
            "recall_malware": MODEL_CONFIG["recall_malware"],
        },
        "confusion_matrix": MODEL_CONFIG["confusion_matrix"],
        "top_10_features": MODEL_CONFIG["top_features"],
    }


# =============================================================================
# TAI MODEL ALERT PRIORITY VA CHAY DU DOAN
# =============================================================================

class AlertPriorityModelManager:
    _bundle: Optional[Dict[str, Any]] = None

    @classmethod
    def bundle(cls) -> Dict[str, Any]:
        if cls._bundle is None:
            if not ALERT_PRIORITY_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model not found: {ALERT_PRIORITY_MODEL_PATH}. "
                    f"Train or copy the model to {DEFAULT_ALERT_MODEL_PATH}, "
                    "or set ALERT_PRIORITY_MODEL_PATH to the .joblib file."
                )
            cls._bundle = joblib.load(ALERT_PRIORITY_MODEL_PATH)
        return cls._bundle


def priority_score(priority: str, confidence: float) -> float:
    base = {"P1": 0.95, "P2": 0.82, "P3": 0.55, "P4": 0.20}.get(priority, 0.20)
    return round(min(1.0, max(0.0, (base * 0.75) + (confidence * 0.25))), 4)


PRIORITY_RANK = {"P1": 4, "P2": 3, "P3": 2, "P4": 1}


def higher_priority(left: str, right: str) -> str:
    return left if PRIORITY_RANK.get(left, 0) >= PRIORITY_RANK.get(right, 0) else right


def coerce_alert_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Chap nhan ca body boc boi Shuffle va alert Wazuh raw trong cung endpoint."""
    if isinstance(payload.get("alert"), dict):
        return payload["alert"]
    if isinstance(payload.get("exec"), dict):
        return payload["exec"]
    return payload


def coerce_batch_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts = payload.get("alerts")
    if isinstance(alerts, list):
        return [coerce_alert_payload(item) for item in alerts if isinstance(item, dict)]
    return [coerce_alert_payload(payload)]


def get_nested(data: Dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def extract_block_ip(alert: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
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
    bundle = AlertPriorityModelManager.bundle()
    pipeline = bundle["pipeline"]
    metadata = bundle.get("metadata", {})
    features = extract_features(alert)
    model_priority = str(pipeline.predict([features])[0])
    rule_priority = derive_priority_label(alert)
    priority = higher_priority(rule_priority, model_priority)

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
    attack_type = normalize_attack_type(alert)

    if attack_type in {"command_injection", "webshell", "sql_injection", "dos"}:
        priority = higher_priority("P1", priority)
        score = max(score, priority_score(priority, confidence))
        should_notify = True
    elif attack_type in {"xss", "ssh_bruteforce"}:
        priority = higher_priority("P2", priority)
        score = max(score, priority_score(priority, confidence))
        should_notify = True

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


def require_ingest_api_key(header_value: Optional[str]) -> None:
    if not INGEST_API_KEY:
        return
    if header_value != INGEST_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid ML ingest API key")


def post_pfsense_block(block_ip: Optional[str], prediction: Dict[str, Any]) -> Dict[str, Any]:
    if not block_ip:
        return {"success": False, "skipped": True, "reason": "missing_block_ip"}
    if not PFSENSE_API_KEY:
        return {"success": False, "skipped": True, "reason": "pfsense_api_key_not_configured"}

    payload = {
        "id": 0,
        "name": PFSENSE_ALIAS_NAME,
        "type": "host",
        "descr": "Block ips from Wazuh ML service",
        "address": [block_ip],
        "detail": [f"auto blocked by ML service priority={prediction.get('priority', 'unknown')}"],
        "apply": True,
    }
    headers = {
        "X-API-Key": PFSENSE_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        response = requests.patch(
            PFSENSE_ALIAS_URL,
            json=payload,
            headers=headers,
            timeout=PFSENSE_TIMEOUT,
            verify=PFSENSE_VERIFY_SSL,
        )
        body: Any
        try:
            body = response.json()
        except ValueError:
            body = response.text
        return {
            "success": response.ok,
            "status_code": response.status_code,
            "body": body,
        }
    except Exception as exc:
        logger.warning(f"[!] pfSense block failed: {exc}")
        return {"success": False, "error": str(exc)}


def telegram_alert_text(alert: Dict[str, Any], prediction: Dict[str, Any], pfsense_result: Dict[str, Any]) -> str:
    rule = alert.get("rule") if isinstance(alert.get("rule"), dict) else {}
    agent = alert.get("agent") if isinstance(alert.get("agent"), dict) else {}
    data = alert.get("data") if isinstance(alert.get("data"), dict) else {}
    pfsense_status = (
        "skipped"
        if pfsense_result.get("skipped")
        else str(pfsense_result.get("status_code", "error"))
    )

    return (
        "Web/IDS Attack Detection Alert\n\n"
        f"Priority: {prediction.get('priority')}\n"
        f"Score: {prediction.get('score')}\n"
        f"Confidence: {prediction.get('confidence')}\n"
        f"Should notify: {prediction.get('should_notify')}\n\n"
        f"Rule ID: {rule.get('id', '')}\n"
        f"Rule level: {rule.get('level', '')}\n"
        f"Description: {rule.get('description', '')}\n\n"
        f"Source IP: {prediction.get('block_ip') or data.get('srcip', '')}\n"
        f"Destination: {data.get('dstip') or data.get('dest_ip') or data.get('dst_ip') or ''}\n"
        f"URL: {data.get('url', '')}\n"
        f"Signature ID: {data.get('id', '')}\n\n"
        f"Agent: {agent.get('name', '')} ({agent.get('ip', '')})\n"
        f"Location: {alert.get('location', '')}\n"
        f"pfSense status: {pfsense_status}\n"
        f"Block source: {prediction.get('block_ip_source')}"
    )


def post_telegram_alert(alert: Dict[str, Any], prediction: Dict[str, Any], pfsense_result: Dict[str, Any]) -> Dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return {"success": False, "skipped": True, "reason": "telegram_not_configured"}

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": telegram_alert_text(alert, prediction, pfsense_result),
    }
    try:
        response = requests.post(url, json=payload, timeout=TELEGRAM_TIMEOUT)
        body: Any
        try:
            body = response.json()
        except ValueError:
            body = response.text
        return {
            "success": response.ok,
            "status_code": response.status_code,
            "body": body,
        }
    except Exception as exc:
        logger.warning(f"[!] Telegram notification failed: {exc}")
        return {"success": False, "error": str(exc)}


def post_shuffle_response(alert: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    if not SHUFFLE_RESPONSE_WEBHOOK:
        return {"success": False, "skipped": True, "reason": "shuffle_webhook_not_configured"}

    payload = {
        "alert": alert,
        "prediction": prediction,
        "pipeline": {
            "source": "wazuh",
            "stage": "ml_service",
            "next": "shuffle_pfsense_telegram",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        response = requests.post(
            SHUFFLE_RESPONSE_WEBHOOK,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=SHUFFLE_RESPONSE_TIMEOUT,
            verify=SHUFFLE_RESPONSE_VERIFY_SSL,
        )
        body: Any
        try:
            body = response.json()
        except ValueError:
            body = response.text
        return {
            "success": response.ok,
            "status_code": response.status_code,
            "body": body,
        }
    except Exception as exc:
        logger.warning(f"[!] Shuffle response webhook failed: {exc}")
        return {"success": False, "error": str(exc)}


def handle_wazuh_ingest(alert: Dict[str, Any]) -> Dict[str, Any]:
    prediction = predict_one(alert)
    priority = str(prediction.get("priority", "")).upper()
    should_notify = bool(prediction.get("should_notify")) or priority in INGEST_NOTIFY_PRIORITIES

    shuffle_result = {"success": False, "skipped": True, "reason": "priority_not_notified"}
    if should_notify:
        shuffle_result = post_shuffle_response(alert, prediction)

    return {
        "prediction": prediction,
        "actions": {
            "shuffle": shuffle_result,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def alert_priority_model_info_payload() -> Dict[str, Any]:
    try:
        bundle = AlertPriorityModelManager.bundle()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return bundle.get("metadata", {})


# =============================================================================
# KHOI TAO FASTAPI APP VA CAC ENDPOINT
# =============================================================================

app = FastAPI(
    title="Combined Malware Detection and Alert Priority ML Service",
    description="Combined FastAPI service for EMBER2024 malware detection and Wazuh alert priority inference.",
    version="1.0.0",
)


@app.get("/health")
def health() -> Dict[str, Any]:
    malware_loaded = MalwareModelManager._model is not None
    return {
        "status": "ok",
        "malware_detection": {
            "status": "ok" if malware_loaded else "model_not_loaded",
            "model_path": MODEL_CONFIG["model_path"],
            "model_exists": os.path.isfile(
                os.path.abspath(os.path.expandvars(os.path.expanduser(MODEL_CONFIG["model_path"])))
            ),
            "model_loaded": malware_loaded,
        },
        "alert_priority": {
            "status": "ok",
            "model_exists": ALERT_PRIORITY_MODEL_PATH.exists(),
            "model_path": str(ALERT_PRIORITY_MODEL_PATH),
        },
        "python": sys.executable,
    }


@app.get("/malware/health")
def malware_health() -> Dict[str, Any]:
    loaded = MalwareModelManager._model is not None
    return {
        "status": "ok" if loaded else "model_not_loaded",
        "model_path": MODEL_CONFIG["model_path"],
        "model_loaded": loaded,
    }


@app.get("/alert-priority/health")
def alert_priority_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_exists": ALERT_PRIORITY_MODEL_PATH.exists(),
        "model_path": str(ALERT_PRIORITY_MODEL_PATH),
        "python": sys.executable,
    }


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    return {
        "malware_detection": malware_model_info_payload(),
        "alert_priority": alert_priority_model_info_payload(),
    }


@app.get("/malware/model-info")
def malware_model_info() -> Dict[str, Any]:
    return malware_model_info_payload()


@app.get("/alert-priority/model-info")
def alert_priority_model_info() -> Dict[str, Any]:
    return alert_priority_model_info_payload()


@app.post("/predict-features")
def predict_features(inp: PEInput):
    try:
        label, probability, threshold, detection_reason, rule_hits = run_inference(inp.features, inp.path)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    result = PredictionResult(
        path=inp.path,
        label=label,
        probability=round(probability, 6),
        threshold=threshold,
        detection_reason=detection_reason,
        rule_hits=rule_hits,
        agent_name=inp.agent_name,
        agent_ip=inp.agent_ip,
        timestamp=datetime.now().isoformat(),
    )

    log_to_csv(result)

    if label == "malicious":
        logger.warning(
            f"[!] MALWARE detected | path={inp.path} "
            f"prob={probability:.4f} agent={inp.agent_name}@{inp.agent_ip}"
        )
        notify_shuffle(result)
    else:
        logger.info(
            f"[+] Legitimate | path={inp.path} "
            f"prob={probability:.4f} agent={inp.agent_name}@{inp.agent_ip}"
        )

    return result.dict()


@app.post("/predict", response_model=PriorityPrediction)
def predict(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    try:
        return predict_one(coerce_alert_payload(payload))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict-batch")
def predict_batch(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    try:
        return {"results": [predict_one(alert) for alert in coerce_batch_payload(payload)]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest/wazuh")
def ingest_wazuh(
    payload: Dict[str, Any] = Body(...),
    x_ml_api_key: Optional[str] = Header(default=None, alias="X-ML-API-Key"),
) -> Dict[str, Any]:
    require_ingest_api_key(x_ml_api_key)
    try:
        return handle_wazuh_ingest(coerce_alert_payload(payload))
    except Exception as exc:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.on_event("startup")
async def startup_event():
    try:
        MalwareModelManager.get()
    except FileNotFoundError as e:
        logger.error(f"[!] {e}")
        logger.warning(
            "[!] Server will run, but /predict-features will return 503 until the malware model path is fixed."
        )


def parse_ports(value: str) -> List[int]:
    ports: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        port = int(item)
        if port not in ports:
            ports.append(port)
    if not ports:
        raise ValueError("No ports configured")
    return ports


def run_uvicorn_port(host: str, port: int) -> None:
    import uvicorn

    uvicorn.run("combined_ml_server:app", host=host, port=port)


if __name__ == "__main__":
    import multiprocessing

    host = os.getenv("COMBINED_ML_HOST", "0.0.0.0")
    ports_env = os.getenv("COMBINED_ML_PORTS", "").strip()
    if ports_env:
        ports = parse_ports(ports_env)
    elif os.getenv("COMBINED_ML_PORT"):
        ports = [int(os.getenv("COMBINED_ML_PORT", "8000"))]
    else:
        ports = [8000, 8010]

    if len(ports) == 1:
        run_uvicorn_port(host, ports[0])
    else:
        processes = []
        logger.info(f"[+] Starting combined ML service on {host} ports: {ports}")
        for port in ports:
            proc = multiprocessing.Process(
                target=run_uvicorn_port,
                args=(host, port),
                name=f"combined-ml-port-{port}",
            )
            proc.start()
            processes.append(proc)

        try:
            for proc in processes:
                proc.join()
        except KeyboardInterrupt:
            logger.info("[+] Stopping combined ML service processes")
            for proc in processes:
                proc.terminate()
            for proc in processes:
                proc.join(timeout=5)
