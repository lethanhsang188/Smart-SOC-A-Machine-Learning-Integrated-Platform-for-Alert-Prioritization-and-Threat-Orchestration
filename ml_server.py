"""
ml_server.py
=============
FastAPI ML server — Load EMBER2024 LightGBM model, nhận feature vector
từ pe_extractor_service.py, dự đoán malware, gửi cảnh báo sang Shuffle.

Model metadata (từ training_results_20260416_083422.pkl / .json):
  model_path   : E:\\ML Service\\ember2024_pe_lgbm.model
  num_trees    : 999
  num_features : 2560
  auc_roc      : 0.9983
  fpr          : 1.23%   (FP rate trên test set)
  fnr          : 2.64%   (FN rate trên test set)
  support      : 540,000 benign + 540,000 malware

Chạy:
    pip install fastapi uvicorn lightgbm pandas
    uvicorn ml_server:app --host 0.0.0.0 --port 8000

Endpoint:
    POST /predict-features    ← nhận từ pe_extractor_service
    GET  /health
    GET  /model-info
"""

import os
import traceback
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
import urllib3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import lightgbm as lgb
except ImportError:
    raise RuntimeError("pip install lightgbm")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "ember2024_pe_lgbm.model"
DEFAULT_LOG_CSV = BASE_DIR / "logs" / "malware_logs.csv"

# =============================================================================
# MODEL CONFIG  —  giữ nguyên giá trị từ pkl / json đã upload
# =============================================================================

MODEL_CONFIG = {
    # Windows default: put ember2024_pe_lgbm.model next to this file.
    # Override with ML_MODEL_PATH when running as a service.
    "model_path":    os.getenv("ML_MODEL_PATH", str(DEFAULT_MODEL_PATH)),

    # Metadata từ model_info trong pkl
    "num_trees":     999,
    "num_features":  2560,
    "best_iteration": -1,           # lgb dùng tất cả 999 cây

    # Metrics trên test set (540K benign + 540K malware) từ pkl
    "auc_roc":            0.9983184980315499,
    "accuracy":           0.9806407407407407,
    "fpr":                0.012303703703703704,   # False Positive Rate
    "fnr":                0.026414814814814815,   # False Negative Rate
    "f1_malware":         0.9805031798429661,
    "f1_benign":          0.9807763741012486,
    "precision_malware":  0.9875201923438146,
    "recall_malware":     0.9735851851851852,

    # Confusion matrix từ pkl (test set)
    "confusion_matrix": {
        "TN": 533356, "FP": 6644,
        "FN": 14264,  "TP": 525736,
    },

    # Categorical feature indices (từ model.py)
    # idx 2=is_pe, 3-6=start_bytes, 701=machine_type, 702=subsystem
    "categorical_feature_indices": [2, 3, 4, 5, 6, 701, 702],

    # Top-10 feature importance từ pkl (theo gain)
    "top_features": [
        {"rank": 1,  "feature": "Column_739",  "gain": 12856688.58},
        {"rank": 2,  "feature": "Column_2413", "gain": 6572892.55},
        {"rank": 3,  "feature": "Column_2539", "gain": 1972618.21},
        {"rank": 4,  "feature": "Column_2414", "gain": 1829650.54},
        {"rank": 5,  "feature": "Column_993",  "gain": 956657.88},
        {"rank": 6,  "feature": "Column_509",  "gain": 831880.35},
        {"rank": 7,  "feature": "Column_994",  "gain": 781334.71},
        {"rank": 8,  "feature": "Column_508",  "gain": 738421.92},
        {"rank": 9,  "feature": "Column_2405", "gain": 638726.59},
        {"rank": 10, "feature": "Column_991",  "gain": 615930.90},
    ],
}

# ── Ngưỡng phán quyết ─────────────────────────────────────────────────────────
# Mặc định 0.5 — có thể tăng lên (ví dụ 0.7) để giảm FPR
MALWARE_THRESHOLD = float(os.getenv("ML_MALWARE_THRESHOLD", "0.5"))

# Secondary policy for known offensive tools. This avoids lowering the global
# model threshold while still catching samples such as Win32 mimikatz.exe.
SUSPICIOUS_TOOL_THRESHOLD = float(os.getenv("ML_SUSPICIOUS_TOOL_THRESHOLD", "0.30"))
HIGH_RISK_TOOL_NAMES = {
    "mimikatz.exe",
    "mimidrv.sys",
}
HIGH_RISK_PATH_TOKENS = {
    "mimikatz-master",
}

# ── Shuffle webhooks ───────────────────────────────────────────────────────────
SHUFFLE_MALWARE_WEBHOOK = os.getenv(
    "SHUFFLE_MALWARE_WEBHOOK",
    "",
).strip()
SHUFFLE_VERIFY_SSL = os.getenv("SHUFFLE_VERIFY_SSL", "0").lower() in ("1", "true", "yes")
SHUFFLE_CA_BUNDLE = os.getenv("SHUFFLE_CA_BUNDLE", "").strip()

# ── Log file ───────────────────────────────────────────────────────────────────
LOG_CSV = os.getenv("ML_LOG_CSV", str(DEFAULT_LOG_CSV))

# =============================================================================
# MODEL LOADING
# =============================================================================

class ModelManager:
    """Singleton — load model một lần duy nhất khi khởi động."""
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
            logger.info(f"[+] Loading model: {mp}")
            cls._model = lgb.Booster(model_file=mp)
            logger.info(
                f"[+] Model loaded — trees={MODEL_CONFIG['num_trees']} "
                f"features={MODEL_CONFIG['num_features']} "
                f"AUC={MODEL_CONFIG['auc_roc']:.4f}"
            )
        return cls._model


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="EMBER2024 Malware Detection API",
    description=(
        "EMBER2024 LightGBM — 999 trees, 2560 features\n"
        f"AUC-ROC: {MODEL_CONFIG['auc_roc']:.4f} | "
        f"FPR: {MODEL_CONFIG['fpr']*100:.2f}% | "
        f"FNR: {MODEL_CONFIG['fnr']*100:.2f}%"
    ),
    version="1.0.0",
)


# =============================================================================
# SCHEMAS
# =============================================================================

class PEInput(BaseModel):
    features:   List[float]   # shape (2560,) — từ pe_extractor_service.py
    path:       str
    agent_name: str = ""
    agent_ip:   str = ""


class PredictionResult(BaseModel):
    path:        str
    label:       str          # "malicious" | "legitimate"
    probability: float        # P(malware)
    threshold:   float
    detection_reason: str = ""
    rule_hits: List[str] = Field(default_factory=list)
    agent_name:  str
    agent_ip:    str
    timestamp:   str


# =============================================================================
# INFERENCE
# =============================================================================

def apply_detection_policy(path: str, probability: float) -> tuple[str, float, str, List[str]]:
    """Apply ML threshold plus a narrow high-risk tool policy."""
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
    """
    Nhận list[float] (2560 dims), trả về (label, probability).
    Sử dụng EMBER2024 LightGBM model từ pkl/json.
    """
    model = ModelManager.get()

    n = MODEL_CONFIG["num_features"]
    if len(features) != n:
        raise ValueError(
            f"Feature dimension mismatch: nhận {len(features)}, "
            f"model cần {n}"
        )

    x = np.array(features, dtype=np.float32).reshape(1, -1)
    prob = float(model.predict(x)[0])
    label, threshold, reason, rule_hits = apply_detection_policy(path, prob)
    return label, prob, threshold, reason, rule_hits


# =============================================================================
# LOGGING TO CSV
# =============================================================================

def log_to_csv(result: PredictionResult) -> None:
    """Ghi kết quả vào CSV — tạo header nếu file chưa tồn tại."""
    try:
        os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
        row = {
            "timestamp":   result.timestamp,
            "agent_name":  result.agent_name,
            "agent_ip":    result.agent_ip,
            "path":        result.path,
            "label":       result.label,
            "probability": result.probability,
            "threshold":   result.threshold,
            "detection_reason": result.detection_reason,
            "rule_hits":   ";".join(result.rule_hits),
        }
        df = pd.DataFrame([row])
        write_header = not os.path.isfile(LOG_CSV)
        df.to_csv(LOG_CSV, mode="a", index=False, header=write_header)
    except Exception as e:
        logger.warning(f"[!] CSV logging failed: {e}")


# =============================================================================
# SHUFFLE NOTIFICATION
# =============================================================================

def notify_shuffle(result: PredictionResult) -> None:
    """Gửi cảnh báo malware sang Shuffle SOAR."""
    verify_ssl: bool | str = SHUFFLE_CA_BUNDLE or SHUFFLE_VERIFY_SSL
    if verify_ssl is False:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    payload = {
        "agent_name":  result.agent_name,
        "agent_ip":    result.agent_ip,
        "path":        result.path.replace("\\", "\\\\"),
        "label":       result.label,
        "probability": result.probability,
        "threshold":   result.threshold,
        "detection_reason": result.detection_reason,
        "rule_hits":   result.rule_hits,
        "timestamp":   result.timestamp,

        # Thêm context model từ pkl để analyst biết độ tin cậy
        "model_info": {
            "num_trees":   MODEL_CONFIG["num_trees"],
            "auc_roc":     MODEL_CONFIG["auc_roc"],
            "fpr":         MODEL_CONFIG["fpr"],
            "fnr":         MODEL_CONFIG["fnr"],
            "threshold":   MALWARE_THRESHOLD,
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
        logger.info(f"[+] Shuffle notified — status={resp.status_code}")
    except Exception as e:
        logger.warning(f"[!] Shuffle notification failed: {e}")


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
def health():
    """Kiểm tra model đã load chưa."""
    loaded = ModelManager._model is not None
    return {
        "status":       "ok" if loaded else "model_not_loaded",
        "model_path":   MODEL_CONFIG["model_path"],
        "model_loaded": loaded,
    }


@app.get("/model-info")
def model_info():
    """Trả về metadata của model từ pkl/json đã upload."""
    return {
        "model_path":        MODEL_CONFIG["model_path"],
        "num_trees":         MODEL_CONFIG["num_trees"],
        "num_features":      MODEL_CONFIG["num_features"],
        "threshold":         MALWARE_THRESHOLD,
        "suspicious_tool_threshold": SUSPICIOUS_TOOL_THRESHOLD,
        "categorical_indices": MODEL_CONFIG["categorical_feature_indices"],
        "performance": {
            "auc_roc":           MODEL_CONFIG["auc_roc"],
            "accuracy":          MODEL_CONFIG["accuracy"],
            "fpr":               MODEL_CONFIG["fpr"],
            "fnr":               MODEL_CONFIG["fnr"],
            "f1_malware":        MODEL_CONFIG["f1_malware"],
            "f1_benign":         MODEL_CONFIG["f1_benign"],
            "precision_malware": MODEL_CONFIG["precision_malware"],
            "recall_malware":    MODEL_CONFIG["recall_malware"],
        },
        "confusion_matrix": MODEL_CONFIG["confusion_matrix"],
        "top_10_features":  MODEL_CONFIG["top_features"],
    }


@app.post("/predict-features")
def predict_features(inp: PEInput):
    """
    Nhận feature vector 2560-dim từ pe_extractor_service.py,
    chạy inference với EMBER2024 LightGBM,
    log kết quả và notify Shuffle nếu malicious.
    """
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

    # ── Log tất cả kết quả vào CSV ──────────────────────────────────────────
    log_to_csv(result)

    # ── Chỉ notify Shuffle khi phát hiện malware ────────────────────────────
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


# =============================================================================
# ENTRY POINT
# =============================================================================

# Preload model khi server khởi động để giảm latency request đầu tiên
@app.on_event("startup")
async def startup_event():
    try:
        ModelManager.get()
    except FileNotFoundError as e:
        logger.error(f"[!] {e}")
        logger.warning("[!] Server sẽ chạy nhưng /predict-features sẽ trả lỗi 503 "
                       "cho đến khi model được đặt đúng vị trí.")


# Chạy:
# uvicorn ml_server:app --host 0.0.0.0 --port 8000
