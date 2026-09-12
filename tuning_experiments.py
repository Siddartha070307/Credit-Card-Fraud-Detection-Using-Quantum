"""
QSVC tuning experiments -- testing whether rescaling and circuit config
changes close the gap with classical SVM (which scores 0.99 ROC-AUC).
Uses the exact same real dataset, split, and feature selection as the
locked pipeline -- only the scaling method and feature map config vary.
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

RANDOM_STATE = 42
DATA_DIR = Path(__file__).parent / "data"

# ---------- Replicate the exact same split as the locked pipeline ----------
df = pd.read_csv(DATA_DIR / "creditcard.csv").dropna()
selected_features = (DATA_DIR / "selected_features.txt").read_text().strip().split("\n")

X_full = df[selected_features]
y_full = df["Class"]

X_train_full, X_test_full, y_train_full, y_test = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE
)

# ---------- NEW: rescale to [0, pi] instead of StandardScaler ----------
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_full), columns=selected_features, index=X_train_full.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_full), columns=selected_features, index=X_test_full.index
)

# ---------- Undersample majority class, TRAIN set only (identical logic) ----------
rng = np.random.default_rng(RANDOM_STATE)
fraud_idx = y_train_full[y_train_full == 1].index
legit_idx = y_train_full[y_train_full == 0].index
n_legit_keep = int(len(fraud_idx) * 1.0)
legit_sample = rng.choice(legit_idx, size=min(n_legit_keep, len(legit_idx)), replace=False)
keep_idx = np.concatenate([fraud_idx.to_numpy(), legit_sample])
rng.shuffle(keep_idx)
X_train_bal = X_train_scaled.loc[keep_idx]
y_train_bal = y_train_full.loc[keep_idx]

print(f"[data] train (balanced, rescaled to [0,pi]): {X_train_bal.shape}, "
      f"fraud ratio: {y_train_bal.mean():.3f}")

# ---------- Reconstruct the same 200-row test subsample (identical seed/logic) ----------
rng2 = np.random.default_rng(42)
t_fraud_idx = y_test[y_test == 1].index
t_legit_idx = y_test[y_test == 0].index
n_fraud_keep = 20
n_legit_keep_test = 180
fraud_keep = rng2.choice(t_fraud_idx, size=n_fraud_keep, replace=False)
legit_keep = rng2.choice(t_legit_idx, size=n_legit_keep_test, replace=False)
keep_idx_test = np.concatenate([fraud_keep, legit_keep])

X_sub = X_test_scaled.loc[keep_idx_test]
y_sub = y_test.loc[keep_idx_test]
print(f"[data] test subsample: {X_sub.shape}, fraud count: {y_sub.sum()}")

n_features = len(selected_features)

# ---------- Experiment configs ----------
experiments = [
    {"name": "Baseline (StandardScaler, reps=2, linear) -- reference", "reps": 2, "entanglement": "linear", "skip": True},
    {"name": "Exp 1: [0,pi] rescale, reps=2, linear (isolate rescale effect)", "reps": 2, "entanglement": "linear"},
    {"name": "Exp 2: [0,pi] rescale, reps=2, full entanglement", "reps": 2, "entanglement": "full"},
    {"name": "Exp 3: [0,pi] rescale, reps=1, linear", "reps": 1, "entanglement": "linear"},
]

results = []

for exp in experiments:
    if exp.get("skip"):
        results.append({"name": exp["name"], "roc_auc": 0.5139, "pr_auc": 0.1164, "f1": 0.1681, "note": "(known result, not rerun)"})
        continue

    print(f"\n=== Running: {exp['name']} ===")
    t0 = time.time()
    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=exp["reps"], entanglement=exp["entanglement"])
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=kernel)
    qsvc.fit(X_train_bal.values, y_train_bal.values)
    train_time = time.time() - t0
    print(f"  training took {train_time:.1f}s")

    t0 = time.time()
    y_score = qsvc.decision_function(X_sub.values)
    y_pred = (y_score > 0).astype(int)
    infer_time = time.time() - t0
    print(f"  inference took {infer_time:.1f}s")

    roc = roc_auc_score(y_sub, y_score)
    pr = average_precision_score(y_sub, y_score)
    f1 = f1_score(y_sub, y_pred, zero_division=0)

    print(f"  ROC-AUC: {roc:.4f}  PR-AUC: {pr:.4f}  F1: {f1:.4f}")
    print(classification_report(y_sub, y_pred, digits=4, zero_division=0))

    results.append({"name": exp["name"], "roc_auc": round(roc, 4), "pr_auc": round(pr, 4), "f1": round(f1, 4), "note": ""})

print("\n\n========== SUMMARY ==========")
for r in results:
    print(f"{r['name']}: ROC-AUC={r['roc_auc']}, PR-AUC={r['pr_auc']}, F1={r['f1']} {r['note']}")
print(f"\nClassical SVM reference: ROC-AUC=0.9947, PR-AUC=0.9710, F1=0.9474")
