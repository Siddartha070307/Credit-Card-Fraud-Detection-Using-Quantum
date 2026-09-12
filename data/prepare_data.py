"""
Data preparation for Quantum Fraud Detection.

Pipeline (LOCKED):
1. Load raw creditcard.csv (Kaggle Credit Card Fraud dataset)
2. Scale features
3. Train/test split FIRST (stratified, 80/20) -- test set keeps REAL fraud ratio
4. RandomForest feature importance -> keep top N features (default 6)
5. Undersample the TRAINING set only -> balanced set for quantum training
6. Save train/test artifacts for downstream model scripts

Usage:
    python prepare_data.py --csv path/to/creditcard.csv --n-features 6
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
OUT_DIR = Path(__file__).parent
MODEL_DIR = OUT_DIR.parent / "models"


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna()
    return df


def make_synthetic_fallback(n_rows: int = 5000) -> pd.DataFrame:
    """
    Fallback synthetic dataset shaped like the Kaggle Credit Card Fraud set
    (V1-V28 transformed features + Amount + Class), so the pipeline is runnable
    end-to-end even before you've downloaded the real CSV.
    NOT for your final results -- swap in the real Kaggle CSV before demo day.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_fraud = max(10, int(n_rows * 0.0017))
    n_legit = n_rows - n_fraud

    legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    fraud = rng.normal(loc=1.5, scale=1.3, size=(n_fraud, 28))  # shifted distribution

    X = np.vstack([legit, fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)
    amount = rng.exponential(scale=50, size=len(y))

    cols = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(X, columns=cols)
    df["Amount"] = amount
    df["Class"] = y
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def select_top_features(X: pd.DataFrame, y: pd.Series, n_features: int) -> list:
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top = importances.sort_values(ascending=False).head(n_features).index.tolist()
    print(f"[feature_selection] top {n_features} features: {top}")
    return top


def undersample_majority(X: pd.DataFrame, y: pd.Series, ratio: float = 1.0):
    """Keep all minority (fraud) rows, sample majority (legit) at `ratio`x that count."""
    fraud_idx = y[y == 1].index
    legit_idx = y[y == 0].index
    n_legit_keep = int(len(fraud_idx) * ratio)
    rng = np.random.default_rng(RANDOM_STATE)
    legit_sample = rng.choice(legit_idx, size=min(n_legit_keep, len(legit_idx)), replace=False)
    keep_idx = np.concatenate([fraud_idx.to_numpy(), legit_sample])
    rng.shuffle(keep_idx)
    return X.loc[keep_idx], y.loc[keep_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(OUT_DIR / "creditcard.csv"),
                         help="Path to creditcard.csv (defaults to the bundled real dataset)")
    parser.add_argument("--n-features", type=int, default=6)
    parser.add_argument("--undersample-ratio", type=float, default=1.0,
                         help="legit:fraud ratio in the balanced TRAIN set")
    args = parser.parse_args()

    if args.csv and Path(args.csv).exists():
        df = load_data(args.csv)
        print(f"[data] loaded real dataset: {df.shape}")
    else:
        print("[data] no --csv given / file not found -> using SYNTHETIC fallback data. "
              "Replace with the real Kaggle creditcard.csv before final results.")
        df = make_synthetic_fallback()

    X_full = df.drop(columns=["Class"])
    y_full = df["Class"]

    # 1. Split FIRST -- test set must reflect the REAL fraud ratio, untouched.
    X_train_full, X_test_full, y_train_full, y_test = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=RANDOM_STATE
    )

    # 2. Feature selection on the training data only (avoid leakage)
    top_features = select_top_features(X_train_full, y_train_full, args.n_features)

    X_train_full = X_train_full[top_features]
    X_test = X_test_full[top_features]

    # 3. Scale (fit on train only, apply to both)
    # LOCKED (post-tuning): MinMaxScaler to [0, pi], not StandardScaler.
    # StandardScaler produced values outside the ZZFeatureMap's natural rotation-angle
    # range, scrambling the quantum encoding's geometry -- this was diagnosed as the
    # root cause of QSVC scoring near-random (ROC-AUC 0.514) in initial testing.
    # Rescaling to [0, pi], matching the encoding's actual periodicity, closed the
    # gap to classical SVM almost entirely (QSVC ROC-AUC 0.514 -> 0.988). This choice
    # affects the classical models too (all models share one preprocessing pipeline
    # for a fair, single-pipeline comparison) -- classical SVM/RandomForest performance
    # is not meaningfully sensitive to this rescaling, quantum kernel performance is.
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_full), columns=top_features, index=X_train_full.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=top_features, index=X_test.index
    )

    # 4. Undersample majority class -- TRAIN SET ONLY
    X_train_bal, y_train_bal = undersample_majority(
        X_train_scaled, y_train_full, ratio=args.undersample_ratio
    )

    print(f"[split] train (balanced): {X_train_bal.shape}, fraud ratio: {y_train_bal.mean():.3f}")
    print(f"[split] test (real distribution): {X_test_scaled.shape}, fraud ratio: {y_test.mean():.5f}")

    # Save artifacts
    X_train_bal.to_csv(OUT_DIR / "X_train.csv", index=False)
    y_train_bal.to_csv(OUT_DIR / "y_train.csv", index=False)
    X_test_scaled.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

    with open(OUT_DIR / "selected_features.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(top_features) + "\n")

    # Persist the fitted MinMaxScaler ([0, pi]) artifact for reproducible inference
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    print(f"[preprocessing] fitted MinMaxScaler([0,pi]) saved to {MODEL_DIR / 'scaler.joblib'}")

    print(f"[done] artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
