"""
Classical baselines: SVM (the direct comparison for QSVC) and RandomForest
(secondary reference). Trains on the balanced train set, evaluates on the
untouched, real-distribution test set.
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent


def load_splits():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    return X_train, y_train, X_test, y_test


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_score = model.decision_function(X_test)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_score):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, y_score):.4f}")


def main():
    X_train, y_train, X_test, y_test = load_splits()

    # Classical SVM -- THE direct comparison point for QSVC (same kernel family idea)
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    evaluate("Classical SVM (RBF kernel)", svm, X_test, y_test)
    joblib.dump(svm, MODEL_DIR / "classical_svm.joblib")

    # RandomForest -- secondary reference baseline
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    evaluate("RandomForest (secondary baseline)", rf, X_test, y_test)
    joblib.dump(rf, MODEL_DIR / "random_forest.joblib")

    print(f"\n[done] models saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
