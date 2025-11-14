# train.py
# Usage: python train.py
# - Automatically downloads the UCI parkinsons.data file and trains a RandomForest.
# - Optionally will include a local CSV at data/parkinsons_extra.csv if you downloaded the second dataset manually.

import os
import io
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
OUT_MODEL = "parkinsons_rf_pipeline.joblib"
EXTRA_CSV_PATH = os.path.join("data", "parkinsons_extra.csv")  # optional - if you downloaded the second dataset here

def download_uci(out_path="parkinsons.data"):
    if os.path.exists(out_path):
        print(f"[+] Using existing {out_path}")
        return out_path
    print("[*] Downloading UCI Parkinson's dataset...")
    r = requests.get(UCI_URL)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    print("[+] Saved", out_path)
    return out_path

def load_uci(path="parkinsons.data"):
    df = pd.read_csv(path)
    # UCI dataset: column 'status' is label (1 = PD, 0 = healthy)
    return df

def load_extra_if_exists(path=EXTRA_CSV_PATH):
    if os.path.exists(path):
        print("[*] Found extra dataset at", path, "— attempting to load.")
        try:
            extra = pd.read_csv(path)
            print("[+] Extra dataset columns:", list(extra.columns)[:10], "...")
            return extra
        except Exception as e:
            print("[!] Failed to read extra CSV:", e)
            return None
    else:
        print("[i] No extra CSV found at", path)
        return None

def prepare_training_dataframe(uci_df, extra_df=None):
    # We want a combined df with the same features as UCI if possible.
    # UCI has 'name' and 'status' columns; features are all others.
    uci_features = [c for c in uci_df.columns if c not in ("name", "status")]
    print("[i] UCI feature count:", len(uci_features))

    # Start with UCI
    df_comb = uci_df.copy()

    if extra_df is not None:
        # Try to find overlapping feature columns between extra_df and UCI feature names.
        common_cols = [c for c in uci_features if c in extra_df.columns]
        if len(common_cols) < 5:
            print("[!] Only", len(common_cols), "common columns found between UCI and extra dataset.")
            print("[!] We'll only use the UCI dataset to avoid feature mismatch.")
        else:
            print("[+] Found", len(common_cols), "common feature columns. Merging extra data.")
            # We need a label column in extra_df; try to find it
            label_col = None
            for candidate in ("status", "label", "class", "target"):
                if candidate in extra_df.columns:
                    label_col = candidate
                    break
            if label_col is None:
                print("[!] Extra dataset has no obvious label column. Skipping it.")
            else:
                extra_sub = extra_df[common_cols + [label_col]].copy()
                # rename extra label to 'status' for consistency
                extra_sub = extra_sub.rename(columns={label_col: "status"})
                # some datasets might store labels as strings, coerce to int
                extra_sub["status"] = extra_sub["status"].astype(int)
                # Reorder columns to match UCI features order
                extra_sub = extra_sub[uci_features + ["status"]]
                # Append
                df_comb = pd.concat([df_comb[uci_features + ["status"]], extra_sub], ignore_index=True)
                print("[+] Combined dataset size:", df_comb.shape)
    else:
        df_comb = df_comb[uci_features + ["status"]]

    return df_comb

def train_and_save(df):
    X = df.drop(columns=["status"])
    y = df["status"].astype(int)

    # scale features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)

    # model
    clf = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # eval
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    print("\n--- Evaluation on held-out test ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
    except Exception:
        print("ROC AUC: N/A (maybe single-class in test?)")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # cross-validated AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        print("5-fold ROC AUC mean:", cv_scores.mean())
    except Exception as e:
        print("CV failed:", e)

    # save pipeline
    joblib.dump({"model": clf, "scaler": scaler}, OUT_MODEL)
    print("\n[+] Saved model and scaler to", OUT_MODEL)

if __name__ == "__main__":
    uci_path = download_uci()
    uci_df = load_uci(uci_path)
    extra = load_extra_if_exists()
    combined = prepare_training_dataframe(uci_df, extra)
    train_and_save(combined)
