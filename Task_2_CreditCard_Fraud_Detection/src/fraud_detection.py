import warnings
warnings.filterwarnings("ignore")

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support
)

from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
DATA_DIR = TASK_DIR / "data"

TRAIN_PATH = DATA_DIR / "fraudTrain.csv"
TEST_PATH = DATA_DIR / "fraudTest.csv"

# CONFIG

KEY_COLUMNS = [
    "trans_date_trans_time", "amt", "category", "merchant", "state",
    "zip", "lat", "long", "city_pop", "gender",
    "merch_lat", "merch_long", "is_fraud"
]

MAX_TRAIN_ROWS = 300_000
LOAD_TEST_SET = False         
MAX_TEST_ROWS = 150_000       

# OPTIMIZED DATA LOADING

def load_optimized_csv(filepath: Path, is_train=True, max_rows=None) -> pd.DataFrame:
    dtype_dict = {
        "trans_date_trans_time": str,
        "amt": float,
        "category": str,
        "merchant": str,
        "state": str,
        "zip": int,
        "lat": float,
        "long": float,
        "city_pop": int,
        "gender": str,
        "merch_lat": float,
        "merch_long": float,
        "is_fraud": int
    }

    chunksize = 50_000
    chunks = []
    rows_read = 0

    reader = pd.read_csv(
        filepath,
        dtype=dtype_dict,
        usecols=KEY_COLUMNS,
        chunksize=chunksize
    )

    for chunk in reader:
        chunks.append(chunk)
        rows_read += len(chunk)

        if max_rows and rows_read >= max_rows:
            break

    df = pd.concat(chunks, ignore_index=True)
    return df

# FEATURE ENGINEERING
def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")

        df["trans_year"] = df["trans_date_trans_time"].dt.year
        df["trans_month"] = df["trans_date_trans_time"].dt.month
        df["trans_day"] = df["trans_date_trans_time"].dt.day
        df["trans_hour"] = df["trans_date_trans_time"].dt.hour
        df["trans_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek

        df.drop(columns=["trans_date_trans_time"], inplace=True)

    return df


def encode_categorical(df: pd.DataFrame, cat_cols):
    df = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

# REPORTING
def print_clean_report(title, model_name, train_size, val_size,
                       fraud_ratio, roc_auc, pr_auc,
                       y_val, y_val_pred, cm):
    prec1, rec1, f11, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average=None, labels=[0, 1]
    )

    #print("\n" + "=" * 72)
    print("CODSOFT INTERNSHIP - TASK 2")
    print("CREDIT CARD FRAUD DETECTION")
    #print("=" * 72)

    print("\n Dataset Summary")
   # print("-" * 72)
    print(f"Train rows used      : {train_size:,}")
    print(f"Validation rows      : {val_size:,}")
    print(f"Fraud ratio (train)  : {fraud_ratio:.4f}%")

    print("\n Feature Pipeline")
    #print("-" * 72)
    print("Preprocessing        : Label Encoding (categorical) + Standard Scaling (numeric)")
    print("Feature Engineering  : Date-time expansion (year/month/day/hour/dayofweek)")

    print("\n Model")
    #print("-" * 72)
    print(f"Classifier           : {model_name}")
    print("Imbalance Handling   : class_weight='balanced'")

    print("\n Validation Performance")
    #print("-" * 72)
    print(f"ROC-AUC              : {roc_auc:.4f}")
    print(f"PR-AUC               : {pr_auc:.4f}")

    print("\nFraud-class Metrics (Class = 1)")
    #print("-" * 72)
    print(f"Precision (Fraud)    : {prec1[1]:.4f}")
    print(f"Recall (Fraud)       : {rec1[1]:.4f}")
    print(f"F1-score (Fraud)     : {f11[1]:.4f}")

    print("\nConfusion Matrix [TN FP; FN TP]")
    #print("-" * 72)
    print(cm)

    print("\n Classification Report")
    #print("-" * 72)
    print(classification_report(y_val, y_val_pred, digits=4))

    print("\n Execution completed successfully.")
   # print("=" * 72)

# MAIN
def main():
    # Load training dataset (optimized)
    train_df = load_optimized_csv(TRAIN_PATH, is_train=True, max_rows=MAX_TRAIN_ROWS)
    train_df = basic_feature_engineering(train_df)

    # Load test dataset (optional)
    test_df = None
    if LOAD_TEST_SET:
        test_df = load_optimized_csv(TEST_PATH, is_train=False, max_rows=MAX_TEST_ROWS)
        test_df = basic_feature_engineering(test_df)

    if "is_fraud" not in train_df.columns:
        raise ValueError("Target column 'is_fraud' not found.")

    y_full = train_df["is_fraud"]
    X_full = train_df.drop(columns=["is_fraud"])

    # Identify columns
    cat_cols = X_full.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_full.columns if c not in cat_cols]

    # Encode categories
    X_full = encode_categorical(X_full, cat_cols)

    # Train / Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", "passthrough", cat_cols)
        ]
    )

    model = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        n_jobs=1,
        solver="lbfgs",
        tol=1e-3
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model)
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Validate
    y_val_pred = pipe.predict(X_val)
    y_val_proba = pipe.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, y_val_proba)
    pr_auc = average_precision_score(y_val, y_val_proba)
    cm = confusion_matrix(y_val, y_val_pred)

    fraud_ratio = (y_full.mean() * 100)

    # Clean final report
    print_clean_report(
        title="Fraud Detection",
        model_name="Logistic Regression",
        train_size=len(X_train),
        val_size=len(X_val),
        fraud_ratio=fraud_ratio,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        y_val=y_val,
        y_val_pred=y_val_pred,
        cm=cm
    )

    if test_df is not None and "is_fraud" in test_df.columns:
        print("\nNOTE: Test dataset loaded for inference/evaluation.")
    else:
        print("\nNOTE: fraudTest.csv evaluation is disabled (LOAD_TEST_SET=False).")


if __name__ == "__main__":
    main()

