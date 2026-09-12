"""
LangGraph pipeline: Ingest -> Quantum-Score -> Classical-Score -> Decide/Report

Orchestrated 4-node agent architecture for Quantum Fraud Detection (QAIC UC016).
The FastAPI endpoint in api/main.py provides low-latency direct evaluation,
while this graph version allows inspecting the multi-step agent reasoning trace
in the dashboard and CLI.

Report node uses Claude Sonnet for polished fraud analyst explanations,
with an automatic template-string fallback when API access is absent.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import TypedDict, Optional, List, Dict, Any, Union

import joblib
import numpy as np
import pandas as pd
from langgraph.graph import StateGraph, END

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# ---------- Safe Asset Loading ----------

_features_file = DATA_DIR / "selected_features.txt"
if _features_file.exists():
    selected_features = [line.strip() for line in _features_file.read_text(encoding="utf-8").splitlines() if line.strip()]
else:
    selected_features = ["V17", "V12", "V14", "V10", "V16", "V11"]

_qsvc = None
_classical_svm = None
_scaler = None


def get_selected_features() -> List[str]:
    return selected_features


def get_scaler():
    global _scaler
    if _scaler is None:
        scaler_file = MODEL_DIR / "scaler.joblib"
        if not scaler_file.exists():
            scaler_file = DATA_DIR / "scaler.joblib"
        if scaler_file.exists():
            _scaler = joblib.load(scaler_file)
    return _scaler


def get_models():
    global _qsvc, _classical_svm
    if _qsvc is None:
        qsvc_file = MODEL_DIR / "qsvc_model.joblib"
        if not qsvc_file.exists():
            raise FileNotFoundError(
                f"QSVC model not found at {qsvc_file}. Run 'python models/quantum_model.py' first."
            )
        _qsvc = joblib.load(qsvc_file)

    if _classical_svm is None:
        svm_file = MODEL_DIR / "classical_svm.joblib"
        if not svm_file.exists():
            raise FileNotFoundError(
                f"Classical SVM not found at {svm_file}. Run 'python models/classical_baseline.py' first."
            )
        _classical_svm = joblib.load(svm_file)

    return _qsvc, _classical_svm


# ---------- State Definition ----------

class FraudState(TypedDict):
    features: List[float]
    raw_input: Optional[Union[List[float], Dict[str, float]]]
    scaled: Optional[bool]
    quantum_label: Optional[str]
    quantum_score: Optional[float]
    classical_label: Optional[str]
    classical_score: Optional[float]
    decision: Optional[str]
    explanation: Optional[str]
    node_history: Optional[List[Dict[str, Any]]]


# ---------- Node Implementations ----------

def ingest_node(state: FraudState) -> FraudState:
    """Validate and normalize the input feature vector."""
    t0 = time.time()
    raw = state.get("features")
    if not raw and "raw_input" in state:
        raw = state["raw_input"]

    clean_features: List[float] = []

    if isinstance(raw, dict):
        missing = [f for f in selected_features if f not in raw]
        if missing:
            raise ValueError(
                f"Missing required feature keys in input dict: {missing}. Expected: {selected_features}"
            )
        raw_list = [raw[f] for f in selected_features]
    elif isinstance(raw, (list, tuple)):
        raw_list = list(raw)
    else:
        raise ValueError(
            f"Input 'features' must be a list of {len(selected_features)} numbers or a dict of features."
        )

    if len(raw_list) != len(selected_features):
        raise ValueError(
            f"Expected {len(selected_features)} features ({selected_features}), but received {len(raw_list)}."
        )

    for i, val in enumerate(raw_list):
        try:
            fval = float(val)
        except (ValueError, TypeError):
            raise ValueError(
                f"Feature at index {i} ('{selected_features[i]}') is not numeric: {val}"
            )
        if not math.isfinite(fval):
            raise ValueError(
                f"Feature at index {i} ('{selected_features[i]}') must be a finite number: {val}"
            )
        clean_features.append(fval)

    is_already_scaled = state.get("scaled")
    if is_already_scaled is None:
        is_already_scaled = True

    scaler = get_scaler()
    if not is_already_scaled and scaler is not None:
        raw_df = pd.DataFrame([clean_features], columns=selected_features)
        scaled_vec = scaler.transform(raw_df)[0]
        transformed_features = [round(float(v), 4) for v in scaled_vec]
        desc = f"Standardized 6 raw features using persisted StandardScaler ({', '.join(selected_features)})"
    else:
        transformed_features = clean_features
        desc = f"Validated {len(clean_features)} normalized features ({', '.join(selected_features)})"

    state["features"] = transformed_features
    state["raw_input"] = raw
    state["scaled"] = True

    history = state.get("node_history") or []
    history.append({
        "node": "ingest",
        "description": desc,
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
        "output": {"features": transformed_features},
    })
    state["node_history"] = history
    return state


def quantum_score_node(state: FraudState) -> FraudState:
    """Score transaction using QSVC (ZZFeatureMap quantum kernel)."""
    t0 = time.time()
    qsvc, _ = get_models()

    X = np.array(state["features"]).reshape(1, -1)
    try:
        score = float(qsvc.decision_function(X)[0])
        pred = 1 if score > 0 else 0
    except Exception:
        pred = int(qsvc.predict(X)[0])
        score = float(pred)

    state["quantum_label"] = "FRAUD" if pred == 1 else "LEGIT"
    state["quantum_score"] = round(score, 4)

    history = state.get("node_history") or []
    history.append({
        "node": "quantum_score",
        "description": f"QSVC Kernel eval (6 qubits, ZZFeatureMap) -> Margin: {state['quantum_score']}",
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
        "output": {
            "quantum_label": state["quantum_label"],
            "quantum_score": state["quantum_score"],
        },
    })
    state["node_history"] = history
    return state


def classical_score_node(state: FraudState) -> FraudState:
    """Score transaction using Classical SVM (RBF kernel baseline)."""
    t0 = time.time()
    _, classical_svm = get_models()

    X = pd.DataFrame([state["features"]], columns=selected_features)
    pred = int(classical_svm.predict(X)[0])
    try:
        score = float(classical_svm.predict_proba(X)[0][1])
    except Exception:
        dec = float(classical_svm.decision_function(X)[0])
        score = 1.0 / (1.0 + math.exp(-dec))

    state["classical_label"] = "FRAUD" if pred == 1 else "LEGIT"
    state["classical_score"] = round(score, 4)

    history = state.get("node_history") or []
    history.append({
        "node": "classical_score",
        "description": f"Classical SVM RBF eval -> Fraud Prob: {round(score * 100, 2)}%",
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
        "output": {
            "classical_label": state["classical_label"],
            "classical_score": state["classical_score"],
        },
    })
    state["node_history"] = history
    return state


def _template_explanation(state: FraudState) -> str:
    agreement = "agree" if state["quantum_label"] == state["classical_label"] else "disagree"
    feats = ", ".join(f"{n}={v:.2f}" for n, v in zip(selected_features, state["features"]))
    return (
        f"Quantum model: {state['quantum_label']}. "
        f"Classical model: {state['classical_label']} ({agreement}). "
        f"Based on features [{feats}]."
    )


def decide_and_report_node(state: FraudState) -> FraudState:
    """Apply conservative policy and synthesize analyst explanation."""
    t0 = time.time()
    flagged = (state["quantum_label"] == "FRAUD") or (state["classical_label"] == "FRAUD")
    state["decision"] = "FLAGGED" if flagged else "CLEARED"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    explanation_source = "template_fallback"

    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": (
                        "Write a 1-2 sentence fraud-analyst explanation for a transaction "
                        f"scored by two models. Quantum kernel model says: {state['quantum_label']}. "
                        f"Classical SVM says: {state['classical_label']}. "
                        f"Final decision: {state['decision']}. Be concise and factual, no fluff."
                    ),
                }],
            )
            state["explanation"] = msg.content[0].text.strip()
            explanation_source = "claude_sonnet"
        except Exception as e:
            print(f"[agent] Claude explanation failed ({e}), using template fallback")
            state["explanation"] = _template_explanation(state)
    else:
        state["explanation"] = _template_explanation(state)

    history = state.get("node_history") or []
    history.append({
        "node": "decide_and_report",
        "description": f"Policy decision: {state['decision']} (flag if either flags). Source: {explanation_source}",
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
        "output": {
            "decision": state["decision"],
            "explanation": state["explanation"],
        },
    })
    state["node_history"] = history
    return state


# ---------- Graph Construction ----------

def build_graph():
    graph = StateGraph(FraudState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("quantum_score", quantum_score_node)
    graph.add_node("classical_score", classical_score_node)
    graph.add_node("decide_and_report", decide_and_report_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "quantum_score")
    graph.add_edge("quantum_score", "classical_score")
    graph.add_edge("classical_score", "decide_and_report")
    graph.add_edge("decide_and_report", END)

    return graph.compile()


def run_agent_pipeline_with_trace(
    features: Union[List[float], Dict[str, float]],
    scaled: bool = True
) -> Dict[str, Any]:
    """Execute graph and return comprehensive results with timing and node traces."""
    app = build_graph()
    initial_state: FraudState = {
        "features": [],
        "raw_input": features,
        "scaled": scaled,
        "quantum_label": None,
        "quantum_score": None,
        "classical_label": None,
        "classical_score": None,
        "decision": None,
        "explanation": None,
        "node_history": [],
    }
    t0 = time.time()
    final_state = app.invoke(initial_state)
    total_time = round(time.time() - t0, 3)

    return {
        "decision": final_state["decision"],
        "quantum_label": final_state["quantum_label"],
        "quantum_score": final_state["quantum_score"],
        "classical_label": final_state["classical_label"],
        "classical_score": final_state["classical_score"],
        "explanation": final_state["explanation"],
        "features": final_state["features"],
        "latency_seconds": total_time,
        "trace": final_state.get("node_history", []),
    }


# ---------- CLI Demo Execution ----------

def main():
    parser = argparse.ArgumentParser(description="LangGraph Quantum Fraud Agent Pipeline")
    parser.add_argument(
        "--features",
        nargs="+",
        type=float,
        help=f"List of {len(selected_features)} float values ({selected_features})",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Treat input features as raw values to be transformed with StandardScaler",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["fraud", "fraud2", "legit", "borderline", "zeros"],
        default="fraud2",
        help="Demo preset transaction",
    )
    args = parser.parse_args()

    presets = {
        "fraud": [0.63, -0.81, -0.85, -0.05, 1.35, -0.15],
        "fraud2": [-28.12, -13.60, -13.38, -10.43, -14.03, 6.72],
        "legit": [-0.36, 1.45, 0.23, -1.13, 0.00, 1.88],
        "borderline": [0.17, 1.18, -1.49, 0.14, -0.03, -1.15],
        "zeros": [0.0] * len(selected_features),
    }

    input_features = args.features if args.features else presets[args.preset]

    print("\n" + "=" * 60)
    print("   QAIC UC016: LangGraph Agent Pipeline Execution")
    print("=" * 60)
    mode_str = "Raw (StandardScaler will be applied)" if args.raw else "Normalized"
    print(f"Features ({mode_str}): {dict(zip(selected_features, input_features))}\n")

    result = run_agent_pipeline_with_trace(input_features, scaled=not args.raw)

    print("NODE EXECUTION TRACE:")
    print("-" * 60)
    for step in result["trace"]:
        print(f"[{step['node'].upper()}] ({step['elapsed_ms']}ms): {step['description']}")

    print("-" * 60)
    print(f"FINAL DECISION   : {result['decision']}")
    print(f"QUANTUM QSVC     : {result['quantum_label']} (Margin: {result['quantum_score']:+.4f})")
    print(f"CLASSICAL SVM    : {result['classical_label']} (Prob: {result['classical_score']*100:.1f}%)")
    print(f"EXPLANATION      : {result['explanation']}")
    print(f"TOTAL LATENCY    : {result['latency_seconds']}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
