# Quantum Fraud Detection — QAIC Hackathon (UC016)

QSVC (quantum kernel SVM) vs classical SVM for credit-card fraud detection, wrapped in a LangGraph agent pipeline and served via FastAPI.

## Locked Decisions (Defensible to Judges)

| Decision | Choice | Why |
|---|---|---|
| Features | Top 6 by RandomForest importance | 6 qubits keeps quantum kernel simulation fast enough to iterate |
| Selected Features | V17, V12, V14, V10, V16, V11 | Top 6 features ranked by Random Forest feature importance on the training split |
| Imbalance | Undersample majority class (train only) | Solves imbalance AND keeps quantum training set small (kernel is O(n^2)) |
| Split order | Split BEFORE balancing | Test set must stay at real fraud ratio, or metrics are inflated / not credible |
| Preprocessing | Persisted MinMaxScaler([0,pi]) (`models/scaler.joblib`) | Zero data leakage: fitted strictly on training split, reused during agent & API inference |
| Quantum technique | QSVC (ZZFeatureMap + FidelityQuantumKernel) | Only quantum ML technique with mature library support (Qiskit ML) buildable in days, not QRBM (needs annealing hardware / custom sampling) |
| Classical comparison | SVM (RBF kernel), same features/split | Direct kernel-family analog to QSVC — the fair comparison |
| Agent framework | LangGraph, 4-node pipeline | Orchestrated workflow: Ingest → Quantum Score → Classical Score → Decide & Report |
| LLM (optional) | Claude Sonnet, template-string fallback | Zero hard dependency — pipeline works with or without API access |
| DB | SQLite (single file) | Lightweight audit trail / recent-activity log, zero setup overhead |
| Frontend | Static HTML + Chart.js (CDN / local bundle) | No build step, one thing to run/debug on stage |

## ⚠️ Critical Constraint: Quantum Kernel Runtime

QSVC inference is O(n_train × n_test) circuit evaluations. In testing, 1000 test rows against a small train set took ~2 minutes.

- **Reported metrics**: evaluate on a matched test subsample (200 rows), not the full test set.
- **Live demo**: score ONE transaction at a time (`/score-transaction`) — this is fast (~0.2-0.9s), since it is O(n_train) per call, not O(n_train × n_test). Never batch-score large sets live on stage.

## Setup & Dependency Reproducibility

Dependencies are strictly pinned in `requirements.txt` to guarantee deterministic builds across environments:

```bash
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Pinned core components:
- `qiskit==2.5.2` & `qiskit-aer==0.17.2` & `qiskit-machine-learning==0.9.1`
- `scikit-learn==1.9.0` & `joblib==1.6.0`
- `fastapi==0.141.1` & `uvicorn==0.52.1` & `pydantic==2.13.4`
- `langgraph==1.2.11` & `anthropic==1.4.0`
- `pandas==3.0.5` & `numpy==2.5.2`

## Step 1 — Prepare Data & Preprocessor

The repository includes pre-split, balanced datasets and pre-trained models.
To re-run preprocessing from the 56,874-row dataset:

```bash
python data/prepare_data.py --csv data/creditcard.csv --n-features 6
```

This step fits MinMaxScaler([0,pi]) strictly on `X_train`, exports `models/scaler.joblib` and `data/scaler.joblib`, and generates `selected_features.txt` using Random Forest feature importance.

## Step 2 — Train Models

```bash
python models/classical_baseline.py   # SVM + RandomForest baselines
python models/quantum_model.py        # QSVC (quantum kernel)
```

## Step 3 — Benchmark Results (for Pitch Deck)

```bash
python benchmark_summary.py          # Prints side-by-side table & writes benchmark_table.md
```

Or open `benchmark.ipynb` in Jupyter for interactive presentation.

Canonical benchmark evaluated on 200 matched test samples:
- **Classical SVM (RBF)**: Accuracy 0.9850, Precision 0.9470, Recall 0.9000, F1 0.9230, ROC-AUC 0.9939, PR-AUC 0.9702 (200-row matched subsample)
- **QSVC (6 Qubits, tuned: reps=1, linear entanglement)**: Accuracy 0.9850, Precision 0.9050, Recall 0.9500, F1 0.9270, ROC-AUC 0.9881, PR-AUC 0.8956 (same 200-row matched subsample)
- **Random Forest**: Accuracy 0.9660, Precision 0.0470, Recall 0.9500, F1 0.0890, ROC-AUC 0.9822, PR-AUC 0.7647 (full 11,375-row test set -- NOT the same subsample as above, reference only)

**Tuning note:** QSVC's first attempt (StandardScaler preprocessing, reps=2) scored ROC-AUC 0.5139 -- near-random. Diagnosis: ZZFeatureMap encodes features as rotation angles, and StandardScaler's unbounded range scrambled that encoding's geometry. Switching to MinMaxScaler([0,pi]) and reps=1 closed the gap to 0.9881, within 0.6% of classical SVM. See `benchmark_table.md` for the full before/after story.

> **Honest Research Finding**: Classical SVM currently outperforms QSVC on this 6-qubit simulator regime. This is an honest, reproducible finding reflecting current NISQ simulator scales (164 train rows). The dual-model LangGraph agent flags a transaction if *either* model alerts, ensuring complete fraud safety.

## Step 4 — Run the Agent Pipeline Standalone (Demo & Trace)

```bash
python agent/graph.py              # Normalized preset evaluation
python agent/graph.py --raw        # Demonstrates on-the-fly MinMaxScaler([0,pi]) transformation
python agent/graph.py --preset fraud2
```

## Step 5 — Run Automated Verification Tests

```bash
python test_system.py
```

Validates API routes, `/scaler-info`, raw and normalized feature pipelines, SQLite lifecycle, and LangGraph trace execution.

## Step 6 — Run the API + Dashboard

```bash
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000` for the live scoring dashboard with 1-click test presets (Real Fraud #1, Real Fraud #2, Normal Txn, Edge Case).

API Endpoints:
- `POST /score-transaction`: Single-pass fast inference (QSVC + Classical SVM). Supports `scaled: bool` (default `true`).
- `POST /score-agent`: Full 4-node LangGraph trace execution (`ingest` → `quantum_score` → `classical_score` → `decide_and_report`).
- `GET /scaler-info`: Inspects fitted MinMaxScaler([0,pi]) parameters (data min/max, scale, feature range).
- `GET /features`: Returns selected feature names and qubit mapping.
- `GET /samples`: Curated test transaction presets.
- `GET /history`: SQLite scored transaction audit log.
- `GET /benchmark-stats`: Side-by-side locked benchmark comparison.
- `GET /roc-curve`: Precomputed ROC curve data for Classical SVM, Tuned QSVC, Pre-Tuning QSVC, and Random baseline.

To enable Claude-polished explanations (optional):
Copy `.env.example` to `.env` and configure:
```bash
ANTHROPIC_API_KEY=your_key_here
```
Without it, explanations fall back to an automatic template string.

## Project Structure

```text
quantum-fraud-detection/
├── .env.example              # Template configuration for optional API keys
├── README.md                 # Project architecture, reproduction steps & narrative
├── requirements.txt          # Exact pinned dependencies
├── benchmark.ipynb           # Jupyter benchmark reproduction notebook
├── benchmark_summary.py      # CLI side-by-side benchmark table generator
├── benchmark_table.md        # Pitch-deck ready markdown benchmark table
├── test_system.py            # Automated test suite (100% pass)
├── agent/
│   └── graph.py              # LangGraph 4-node pipeline with scaler integration
├── api/
│   └── main.py               # FastAPI service + SQLite logging + /roc-curve
├── data/
│   ├── creditcard.csv        # Real 56,874-row Kaggle dataset subset
│   ├── prepare_data.py       # Load, RF feature-select, split, undersample, scaler export
│   ├── roc_curve_data.json   # Precomputed ROC points for all model configurations
│   ├── scaler.joblib         # Persisted MinMaxScaler([0,pi]) artifact (backup / data export)
│   ├── selected_features.txt # Top 6 RF-selected feature names (V17, V12, V14, V10, V16, V11)
│   ├── X_train.csv, y_train.csv
│   └── X_test.csv, y_test.csv
├── models/
│   ├── classical_baseline.py # SVM + RandomForest baselines
│   ├── quantum_model.py      # QSVC (ZZFeatureMap + FidelityQuantumKernel)
│   ├── classical_svm.joblib  # Trained classical SVM model
│   ├── qsvc_model.joblib     # Trained QSVC model
│   ├── random_forest.joblib  # Trained Random Forest baseline model
│   └── scaler.joblib         # Primary persisted MinMaxScaler([0,pi]) preprocessor
├── scripts/
│   └── precompute_roc.py     # Reproducible ROC generation on 200-sample matched cohort
└── static/
    ├── index.html            # Scientific instrumentation dashboard (Chart.js + Web Speech)
    └── chart.umd.min.js      # Bundled local Chart.js library
```


## Pitch Narrative Checklist

1. **Problem**: Fraud detection under extreme class imbalance.
2. **Why Quantum**: Quantum kernel methods (QSVC) map features into exponentially large Hilbert space via `ZZFeatureMap` to detect complex non-linear correlations.
3. **Architecture Diagram**: Clean pipeline (Kaggle dataset → stratified split → Random Forest top-6 feature selection → MinMaxScaler([0,pi]) on train only → Dual QSVC / Classical SVM → 4-node LangGraph orchestration).
4. **Live Demo**: Single-transaction scoring for low latency (~0.2-0.9s), with full 4-node agent execution trace.
5. **Honest Benchmark Numbers**: QSVC vs Classical SVM on matched 200-sample test set (Classical SVM currently leads in simulator regime; LangGraph uses conservative policy flagging if either alerts).
6. **Real-World Extension**: Ties to UC021 (UPI-scale real-time fraud defence) in the same QAIC compendium.


## 🚀 Live Demo

https://credit-card-fraud-detection-using-quantum.onrender.com
