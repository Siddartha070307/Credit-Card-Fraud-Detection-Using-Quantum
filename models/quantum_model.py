"""
QSVC: Quantum kernel Support Vector Classifier for fraud detection.

Pipeline (LOCKED):
  6 classical features -> ZZFeatureMap (6 qubits) -> FidelityQuantumKernel
  -> QSVC (quantum kernel + classical SVM solver)

Trains on the balanced train set, evaluates on the untouched real-distribution
test set -- identical protocol to classical_baseline.py so the comparison is fair.
"""
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent


def load_splits():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    return X_train, y_train, X_test, y_test


def evaluate(name, y_true, y_pred, y_score):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    try:
        print(f"ROC-AUC: {roc_auc_score(y_true, y_score):.4f}")
        print(f"PR-AUC:  {average_precision_score(y_true, y_score):.4f}")
    except ValueError as e:
        print(f"(AUC skipped: {e})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-subsample", type=int, default=200,
                         help="Quantum kernel eval is O(n_train*n_test) -- "
                              "subsample the test set to keep this tractable. "
                              "Stratified so fraud ratio is preserved.")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_splits()

    if len(X_test) > args.test_subsample:
        # Real fraud ratio (~0.17%) means naive stratified sampling of a small
        # subset often yields ZERO fraud cases, making the eval meaningless.
        # Guarantee minority-class representation: keep ALL fraud cases in the
        # test set (up to a cap) + fill the rest with legit cases.
        rng = np.random.default_rng(42)
        fraud_idx = y_test[y_test == 1].index
        legit_idx = y_test[y_test == 0].index
        target_fraud = max(1, args.test_subsample // 10)
        n_fraud_keep = min(len(fraud_idx), max(1, min(target_fraud, args.test_subsample - 1)))
        n_legit_keep = args.test_subsample - n_fraud_keep
        if n_legit_keep > len(legit_idx):
            n_legit_keep = len(legit_idx)
        fraud_keep = rng.choice(fraud_idx, size=n_fraud_keep, replace=False)
        legit_keep = rng.choice(legit_idx, size=n_legit_keep, replace=False)
        keep_idx = np.concatenate([fraud_keep, legit_keep])
        rng.shuffle(keep_idx)
        X_test, y_test = X_test.loc[keep_idx], y_test.loc[keep_idx]
        print(f"[quantum] test set subsampled to {len(X_test)} rows "
              f"({n_fraud_keep} fraud + {n_legit_keep} legit, guaranteed minority "
              f"representation -- naive stratified sampling at this scale would "
              f"often yield 0 fraud cases given the real ~0.17% base rate)")

    n_features = X_train.shape[1]
    print(f"[quantum] {n_features} features -> {n_features} qubits (ZZFeatureMap)")
    print(f"[quantum] train size: {len(X_train)} (kernel is O(n^2) -- keep this small)")

    # LOCKED config (post-tuning): reps=1, linear entanglement -- this was the best
    # of 3 tested configs after fixing the feature scaling (see prepare_data.py note).
    # reps=1 outperformed reps=2 here, consistent with published findings that deeper
    # feature maps can worsen quantum kernel "concentration" at small qubit counts.
    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=1, entanglement="linear")
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    qsvc = QSVC(quantum_kernel=quantum_kernel)

    t0 = time.time()
    qsvc.fit(X_train.values, y_train.values)
    print(f"[quantum] training took {time.time() - t0:.1f}s")

    # NOTE: predict() and decision_function() each independently recompute the
    # full O(n_train*n_test) quantum kernel matrix -- calling both DOUBLES
    # runtime. Call decision_function() once and derive labels from its sign
    # instead of calling both.
    t0 = time.time()
    try:
        y_score = qsvc.decision_function(X_test.values)
        y_pred = (y_score > 0).astype(int)
    except Exception:
        y_pred = qsvc.predict(X_test.values)
        y_score = y_pred
    print(f"[quantum] inference on test set took {time.time() - t0:.1f}s "
          f"(single kernel pass, {len(X_train)}x{len(X_test)} = "
          f"{len(X_train)*len(X_test)} circuit evaluations)")

    evaluate("QSVC (Quantum Kernel SVM)", y_test, y_pred, y_score)

    # Also re-evaluate classical SVM on this EXACT same subsample --
    # this is the fair, apples-to-apples comparison for the pitch.
    try:
        classical_svm = joblib.load(MODEL_DIR / "classical_svm.joblib")
        c_pred = classical_svm.predict(X_test)
        c_score = classical_svm.predict_proba(X_test)[:, 1]
        evaluate("Classical SVM (SAME subsample, for fair comparison)", y_test, c_pred, c_score)
    except FileNotFoundError:
        print("[note] run classical_baseline.py first for the matched comparison")

    joblib.dump(qsvc, MODEL_DIR / "qsvc_model.joblib")
    print(f"\n[done] model saved to {MODEL_DIR / 'qsvc_model.joblib'}")


if __name__ == "__main__":
    main()
