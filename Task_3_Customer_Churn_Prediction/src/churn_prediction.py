import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# PATH CONFIG (works from any folder)
SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
DATA_DIR = TASK_DIR / "data"

DATASET_PATH = DATA_DIR / "Churn_Modelling.csv"

# MAIN
def main():
    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Target column
    if "Exited" not in df.columns:
        raise ValueError("Target column 'Exited' not found in Churn_Modelling.csv")

    # Drop ID-like columns 
    drop_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Separate features and target
    y = df["Exited"].astype(int)
    X = df.drop(columns=["Exited"])

    # Fix dtypes for numeric-like strings
    for col in X.columns:
        if X[col].dtype == object:
            tmp = pd.to_numeric(X[col].astype(str).str.replace(',', '').str.strip(), errors="coerce")
            if tmp.notna().sum() / len(tmp) >= 0.5:
                X[col] = tmp

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Impute missing values
    for col in num_cols:
        # median of numeric column (skip if all NaN)
        if X[col].notna().any():
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(0)

    for col in cat_cols:
        if X[col].dtype == object:
            mode = X[col].mode()
            fill_val = mode[0] if not mode.empty else "Unknown"
            X[col] = X[col].fillna(fill_val)
        else:
           mode = X[col].mode()
           fill_val = mode[0] if not mode.empty else 0
           X[col] = X[col].fillna(fill_val)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Models to compare
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    }

    best_model_name = None
    best_pipe = None
    best_auc = -1

    # Train and select best model using ROC-AUC
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        y_val_pred = pipe.predict(X_val)
        y_val_proba = pipe.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, y_val_proba)

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_pipe = pipe

    # Final evaluation
    y_val_pred = best_pipe.predict(X_val)
    y_val_proba = best_pipe.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_proba)
    cm = confusion_matrix(y_val, y_val_pred)

    churn_rate = y.mean() * 100

    # Internship-level output
    #print("\n" + "=" * 72)
    print("CODSOFT INTERNSHIP - TASK 3")
    print("CUSTOMER CHURN PREDICTION")
    #print("=" * 72)

    print("\nDataset Summary")
   #print("-" * 72)
    print(f"Total samples       : {len(df):,}")
    print(f"Training samples    : {len(X_train):,}")
    print(f"Validation samples  : {len(X_val):,}")
    print(f"Churn rate          : {churn_rate:.2f}%")
    print("Target              : Exited (1 = churn, 0 = retained)")

    print("\nPreprocessing Pipeline")
    #print("-" * 72)
    print("Categorical encoding: OneHotEncoder(handle_unknown='ignore')")
    print("Numeric scaling     : StandardScaler()")
    print(f"Categorical cols    : {len(cat_cols)}")
    print(f"Numeric cols        : {len(num_cols)}")

    print("\nBest Model Selected")
    #print("-" * 72)
    print(f"Chosen model        : {best_model_name}")

    print("\nValidation Performance")
    #print("-" * 72)
    print(f"Accuracy            : {acc:.4f} ({acc*100:.2f}%)")
    print(f"ROC-AUC             : {auc:.4f}")

    print("\nConfusion Matrix [TN FP; FN TP]")
    #print("-" * 72)
    print(cm)

    print("\nClassification Report")
    #print("-" * 72)
    print(classification_report(y_val, y_val_pred, digits=4))

    # Feature importance (for RandomForest only)
    if best_model_name == "RandomForest":
        model = best_pipe.named_steps["model"]
        ohe = best_pipe.named_steps["prep"].named_transformers_["cat"]
        cat_features = list(ohe.get_feature_names_out(cat_cols))
        feature_names = num_cols + cat_features

        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:12]

        print("\nTop 12 Important Features")
        #print("-" * 72)
        for i in top_idx:
            print(f"{feature_names[i]:<35} : {importances[i]:.6f}")

    # Sample prediction
    sample = X_val.iloc[[0]]
    sample_pred = best_pipe.predict(sample)[0]
    sample_prob = best_pipe.predict_proba(sample)[0][1]

    print("\nSample Prediction")
    #print("-" * 72)
    print(f"Predicted Exited    : {sample_pred} (1=Churn, 0=No Churn)")
    print(f"Churn Probability   : {sample_prob:.4f}")

    print("\nExecution completed successfully.")
   #print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
