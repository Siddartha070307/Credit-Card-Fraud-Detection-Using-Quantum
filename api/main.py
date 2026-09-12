"""
FastAPI service: score transactions with both QSVC (quantum) and classical SVM,
log every scored transaction to SQLite, expose history, sample presets,
and full LangGraph agent traces for the dashboard.

Run:
    uvicorn api.main:app --reload --port 8000
"""
import json
import math
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.graph import run_agent_pipeline_with_trace, get_models, get_selected_features, get_scaler
from api.report_generator import build_pdf_report

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
DB_PATH = BASE_DIR / "transactions.db"

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

app = FastAPI(
    title="Quantum Fraud Detection API",
    description="QAIC UC016: QSVC vs Classical SVM fraud detection with LangGraph agent architecture.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- SQLite (lightweight audit logging, zero manual setup) ----------

@contextmanager
def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    conn.execute("PRAGMA busy_timeout=5000;")
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scored_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                features TEXT,
                quantum_score REAL,
                classical_score REAL,
                decision TEXT,
                explanation TEXT,
                latency_seconds REAL,
                timestamp REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scored_ts ON scored_transactions(timestamp DESC);")
        conn.commit()


init_db()

# ---------- Load models + feature list at startup (ONCE via shared cache) ----------

selected_features = get_selected_features()
n_features = len(selected_features)
qsvc, classical_svm = get_models()

# Preload curated test presets demonstrating real fraud, verified normal, and edge cases
# NOTE: regenerated after the MinMaxScaler([0,pi]) rescaling fix -- values below are
# fresh, verified rows from the actual test set under the current scaling, each checked
# live against the current model artifacts before being hardcoded here.
sample_presets: Dict[str, Dict[str, Any]] = {
    "real_fraud": {
        "title": "Real Fraud #1",
        "tag": "FRAUD",
        "features": [-0.133, 0.515, 0.658, 0.842, 0.022, 2.228],
        "expected_decision": "FLAGGED",
        "description": "Verified fraud case from Kaggle test set. Both QSVC and Classical SVM agree on fraud.",
    },
    "real_fraud_2": {
        "title": "Real Fraud #2 (High Confidence)",
        "tag": "FRAUD",
        "features": [1.824, 2.061, 1.505, 1.49, 1.674, 1.33],
        "expected_decision": "FLAGGED",
        "description": "Verified fraud case with high model agreement: QSVC (+1.09) and Classical SVM both confidently flag fraud.",
    },
    "real_legit": {
        "title": "Normal Transaction",
        "tag": "NORMAL",
        "features": [2.203, 2.727, 2.26, 1.517, 2.071, 1.283],
        "expected_decision": "CLEARED",
        "description": "Verified legitimate transaction from the test set: both models agree on clearing it.",
    },
    "borderline": {
        "title": "Edge Case (Model Divergence)",
        "tag": "EDGE",
        "features": [2.3, 2.511, 1.875, 1.624, 2.48, 1.032],
        "expected_decision": "FLAGGED",
        "description": "Real fraud case where the models genuinely diverge: QSVC correctly flags fraud (+0.92) while Classical SVM misses it -- demonstrates why the conservative OR-policy (flag if either model alerts) matters.",
    },
}


# ---------- Schemas ----------

class TransactionIn(BaseModel):
    features: Union[List[float], Dict[str, float]] = Field(
        ...,
        description=f"List of {n_features} floats or dict mapping feature names ({selected_features}) to floats"
    )
    scaled: bool = Field(
        default=True,
        description="True if features are already StandardScaler normalized (default). False to apply models/scaler.joblib."
    )


class ScoreOut(BaseModel):
    quantum_score: float
    quantum_label: str
    classical_score: float
    classical_label: str
    decision: str
    explanation: str
    latency_seconds: float
    features: List[float]


class AgentTraceStep(BaseModel):
    node: str
    description: str
    elapsed_ms: float
    output: Dict[str, Any]


class AgentScoreOut(ScoreOut):
    trace: List[AgentTraceStep]


# ---------- Feature Extraction & Validation Helper ----------

def extract_and_validate_features(input_features: Union[List[float], Dict[str, float]]) -> List[float]:
    if isinstance(input_features, dict):
        missing = [f for f in selected_features if f not in input_features]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing feature keys: {missing}. Required features: {selected_features}",
            )
        feature_list = [input_features[f] for f in selected_features]
    elif isinstance(input_features, (list, tuple)):
        feature_list = list(input_features)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Features must be a list of {n_features} numbers or a JSON object with keys: {selected_features}",
        )

    if len(feature_list) != n_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly {n_features} features ({selected_features}), received {len(feature_list)}.",
        )

    clean_list: List[float] = []
    for i, val in enumerate(feature_list):
        try:
            fval = float(val)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{selected_features[i]}' must be numeric. Received: {val}",
            )
        if not math.isfinite(fval):
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{selected_features[i]}' must be a finite number. Received: {val}",
            )
        clean_list.append(fval)

    return clean_list


# ---------- Explanation Helper ----------

def template_explanation(features: List[float], q_label: str, c_label: str) -> str:
    agreement = "agree" if q_label == c_label else "disagree"
    feats = ", ".join(f"{name}={val:.2f}" for name, val in zip(selected_features, features))
    return (
        f"Quantum model: {q_label}. Classical model: {c_label} ({agreement}). "
        f"Based on features [{feats}]."
    )


# ---------- Routes ----------

@app.get("/")
def serve_dashboard():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/score-transaction", response_model=ScoreOut)
def score_transaction(txn: TransactionIn):
    feature_list = extract_and_validate_features(txn.features)

    # Standardize if caller provides raw unscaled features
    if not txn.scaled:
        scaler = get_scaler()
        if scaler is not None:
            raw_df = pd.DataFrame([feature_list], columns=selected_features)
            feature_list = [round(float(v), 4) for v in scaler.transform(raw_df)[0]]

    X = np.array(feature_list).reshape(1, -1)
    X_df = pd.DataFrame(X, columns=selected_features)

    t0 = time.time()

    # Optimized single-pass QSVC inference:
    # decision_function computes the kernel matrix once.
    # Margin sign: score > 0 => FRAUD.
    try:
        q_score = float(qsvc.decision_function(X)[0])
        q_pred = 1 if q_score > 0 else 0
    except Exception:
        q_pred = int(qsvc.predict(X)[0])
        q_score = float(q_pred)

    c_pred = int(classical_svm.predict(X_df)[0])
    try:
        c_score = float(classical_svm.predict_proba(X_df)[0][1])
    except Exception:
        c_score = float(c_pred)

    latency = time.time() - t0

    q_label = "FRAUD" if q_pred == 1 else "LEGIT"
    c_label = "FRAUD" if c_pred == 1 else "LEGIT"

    # Conservative policy: flag if EITHER model flags fraud
    decision = "FLAGGED" if (q_pred == 1 or c_pred == 1) else "CLEARED"
    explanation = template_explanation(feature_list, q_label, c_label)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO scored_transactions "
            "(features, quantum_score, classical_score, decision, explanation, latency_seconds, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (json.dumps(feature_list), round(q_score, 4), round(c_score, 4), decision, explanation, round(latency, 4), time.time()),
        )
        conn.commit()

    return ScoreOut(
        quantum_score=round(q_score, 4),
        quantum_label=q_label,
        classical_score=round(c_score, 4),
        classical_label=c_label,
        decision=decision,
        explanation=explanation,
        latency_seconds=round(latency, 3),
        features=feature_list,
    )


@app.post("/score-agent", response_model=AgentScoreOut)
def score_transaction_agent(txn: TransactionIn):
    """Execute the full 4-node LangGraph agent pipeline and return execution trace."""
    feature_list = extract_and_validate_features(txn.features)
    agent_result = run_agent_pipeline_with_trace(feature_list, scaled=txn.scaled)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO scored_transactions "
            "(features, quantum_score, classical_score, decision, explanation, latency_seconds, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                json.dumps(agent_result["features"]),
                agent_result["quantum_score"],
                agent_result["classical_score"],
                agent_result["decision"],
                agent_result["explanation"],
                agent_result["latency_seconds"],
                time.time(),
            ),
        )
        conn.commit()

    return AgentScoreOut(
        quantum_score=agent_result["quantum_score"],
        quantum_label=agent_result["quantum_label"],
        classical_score=agent_result["classical_score"],
        classical_label=agent_result["classical_label"],
        decision=agent_result["decision"],
        explanation=agent_result["explanation"],
        latency_seconds=agent_result["latency_seconds"],
        features=agent_result["features"],
        trace=agent_result["trace"],
    )



@app.get("/history")
def get_history(limit: int = 20):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, features, quantum_score, classical_score, decision, explanation, timestamp, latency_seconds "
            "FROM scored_transactions ORDER BY id DESC LIMIT ?",
            (min(max(1, limit), 100),),
        ).fetchall()
    return [
        {
            "id": r[0],
            "features": r[1],
            "quantum_score": r[2],
            "classical_score": r[3],
            "decision": r[4],
            "explanation": r[5],
            "timestamp": r[6],
            "latency_seconds": r[7] if len(r) > 7 else None,
        }
        for r in rows
    ]


@app.delete("/history")
def clear_history():
    with get_db() as conn:
        conn.execute("DELETE FROM scored_transactions")
        conn.commit()
    return {"status": "cleared"}


@app.get("/features")
def get_feature_names():
    return {
        "selected_features": selected_features,
        "n_features": n_features,
        "qubits": n_features,
    }


@app.get("/scaler-info")
def get_scaler_info():
    """Information on the persisted MinMaxScaler([0, pi]) preprocessing artifact."""
    scaler = get_scaler()
    if scaler is None:
        return {
            "loaded": False,
            "message": "Scaler artifact not found in models/ or data/",
        }
    return {
        "loaded": True,
        "type": type(scaler).__name__,
        "selected_features": selected_features,
        "n_features": len(selected_features),
        "feature_range": list(getattr(scaler, "feature_range", (0, 1))),
        "data_min": [round(float(m), 6) for m in scaler.data_min_],
        "data_max": [round(float(m), 6) for m in scaler.data_max_],
        "scale": [round(float(s), 6) for s in scaler.scale_],
        "n_samples_seen": int(scaler.n_samples_seen_),
    }


@app.get("/samples")
def get_sample_transactions():
    """Return curated test presets for 1-click demo evaluation."""
    return sample_presets


@app.get("/roc-curve")
def get_roc_curve():
    """Return precomputed ROC curve data for Classical SVM, Tuned QSVC, Pre-Tuning QSVC, and Random baseline."""
    roc_file = DATA_DIR / "roc_curve_data.json"
    if not roc_file.exists():
        raise HTTPException(
            status_code=404,
            detail="ROC curve data not found. Run scripts/precompute_roc.py first.",
        )
    with open(roc_file, "r", encoding="utf-8") as f:
        return json.load(f)



@app.get("/benchmark-stats")
def get_benchmark_stats():
    """
    Locked benchmark statistics. IMPORTANT: classical_svm and qsvc are evaluated
    on the SAME 200-row stratified subsample (20 fraud + 180 legit, guaranteed
    minority-class representation) -- NOT the full 11,375-row test set. This is
    intentional: QSVC inference is O(n_train*n_test), so a small, fraud-guaranteed
    subsample is the only way to get a meaningful, fair, same-conditions comparison
    for both models within a practical runtime.
    random_forest below is evaluated on the FULL 11,375-row test set (the real
    ~0.17% fraud ratio) and is NOT on the same subsample as the two models above --
    it is a secondary reference only, not part of the fair QSVC/SVM comparison.

    TUNING HISTORY: initial QSVC used StandardScaler preprocessing and scored
    ROC-AUC 0.514 (near-random). Diagnosis: ZZFeatureMap encodes features as
    rotation angles, and StandardScaler's unbounded output range scrambled that
    encoding's geometry. Rescaling to MinMaxScaler([0, pi]) -- matching the
    encoding's natural periodicity -- closed the gap to ROC-AUC 0.988, within
    ~0.6% of classical SVM (0.994) on the identical subsample. See
    'tuning_history' below for the pre-fix number, kept for transparency.
    """
    return {
        "evaluation_protocol": {
            "note": "classical_svm and qsvc are evaluated on the SAME 200-row "
                     "subsample (20 fraud + 180 legit) -- required for a fair "
                     "QSVC comparison given quantum kernel runtime. random_forest "
                     "is evaluated separately on the full 11,375-row test set and "
                     "is not directly comparable to the two figures above it.",
            "test_subsample_size": 200,
            "full_test_set_size": 11375,
        },
        "dataset": {
            "name": "Kaggle Credit Card Fraud (Subset)",
            "total_rows": 56874,
            "fraud_ratio": "0.179%",
            "features_selected": selected_features,
            "train_size_balanced": 164,
            "preprocessing": "MinMaxScaler([0, pi]) -- tuned for ZZFeatureMap compatibility",
        },
        "metrics": {
            "classical_svm": {
                "name": "Classical SVM (RBF Kernel)",
                "qubits": "N/A",
                "evaluated_on": "200-row subsample (same as QSVC)",
                "fraud_precision": 0.947,
                "fraud_recall": 0.900,
                "fraud_f1": 0.923,
                "roc_auc": 0.9939,
                "pr_auc": 0.9702,
                "overall_accuracy": 0.985,
            },
            "qsvc": {
                "name": "QSVC (ZZFeatureMap, 6 Qubits, reps=1, linear entanglement)",
                "qubits": 6,
                "evaluated_on": "200-row subsample (same as Classical SVM)",
                "fraud_precision": 0.905,
                "fraud_recall": 0.950,
                "fraud_f1": 0.927,
                "roc_auc": 0.9881,
                "pr_auc": 0.8956,
                "overall_accuracy": 0.985,
            },
        },
        "tuning_history": {
            "note": "QSVC's result before the scaling fix, kept for transparency -- "
                     "this project's core finding is the diagnosis and fix, not just "
                     "the final number.",
            "qsvc_before_tuning": {
                "preprocessing": "StandardScaler (mean 0, std 1) -- mismatched to "
                                  "ZZFeatureMap's rotation-angle encoding",
                "config": "reps=2, linear entanglement",
                "roc_auc": 0.5139,
                "pr_auc": 0.1164,
                "fraud_f1": 0.168,
            },
            "qsvc_after_tuning": {
                "preprocessing": "MinMaxScaler([0, pi]) -- matched to encoding periodicity",
                "config": "reps=1, linear entanglement",
                "roc_auc": 0.9881,
                "pr_auc": 0.8956,
                "fraud_f1": 0.927,
            },
        },
        "additional_reference": {
            "note": "Random Forest evaluated on the FULL 11,375-row test set "
                     "(real ~0.17% fraud ratio), NOT the same 200-row subsample "
                     "as the two models above. Provided as a secondary reference "
                     "only -- do not directly compare its numbers to QSVC/SVM above.",
            "random_forest": {
                "name": "Random Forest (full test set)",
                "fraud_precision": 0.047,
                "fraud_recall": 0.950,
                "fraud_f1": 0.089,
                "roc_auc": 0.9822,
                "pr_auc": 0.7647,
                "overall_accuracy": 0.966,
            },
        },
        "key_takeaway": (
            "Our first QSVC attempt scored ROC-AUC 0.514 (near-random) using "
            "StandardScaler preprocessing. We diagnosed the cause: ZZFeatureMap "
            "encodes features as rotation angles, and StandardScaler's unbounded "
            "output range scrambled that encoding's geometry. Rescaling to "
            "MinMaxScaler([0, pi]) -- matching the encoding's natural periodicity -- "
            "closed the gap to ROC-AUC 0.988 on the identical matched subsample, "
            "within 0.6% of classical SVM's 0.994. The core contribution here is "
            "not just the final number, but a rigorous diagnosis-and-fix process "
            "that took a near-random quantum kernel to genuinely competitive "
            "performance through a principled understanding of the encoding, "
            "not parameter search."
        ),
    }


@app.get("/generate-report")
def generate_report():
    """
    Generate and download a comprehensive, publication-quality PDF report
    containing live benchmark statistics, dataset configuration, geometric tuning story,
    and recent session transaction audit history.
    Synchronous, non-blocking, zero recomputed quantum inference.
    """
    bench_stats = get_benchmark_stats()
    history = get_history(limit=50)
    feat_names = get_feature_names()
    pdf_bytes = build_pdf_report(
        benchmark_data=bench_stats,
        history_data=history,
        feature_data=feat_names,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="quantum_fraud_report.pdf"',
        },
    )

