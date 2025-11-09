"""
Fully automatic PCA reducer for UNSW-NB15.

What it does:
1) Detects and loads the dataset (official train/test or UNSW-NB15_1..4).
2) Drops high-cardinality ID columns.
3) One-hot encodes categorical features.
4) Standardizes all columns.
5) Applies PCA → 4 components (for 4-qubit Q-ADAM).
6) Saves: UNSW_NB15_PCA_4.csv, unsw_scaler.joblib, unsw_pca.joblib.
"""

import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump

# -----------------------------
# Step 1: Locate dataset folder
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "..", "UNSW-NB15 DATASET")
dataset_dir = os.path.abspath(dataset_dir)

if not os.path.isdir(dataset_dir):
    raise FileNotFoundError(f"❌ Could not find dataset folder at: {dataset_dir}")

print(f"📁 Using dataset folder: {dataset_dir}")

# -----------------------------
# Step 2: Load dataset
# -----------------------------
df = None
tt_dir = os.path.join(dataset_dir, "Training and Testing Sets")

if os.path.isdir(tt_dir):
    print("🔍 Checking official train/test CSVs...")
    train_path = next(
        (os.path.join(tt_dir, f) for f in ["UNSW_NB15_training-set.csv", "UNSW-NB15-training.csv"]
         if os.path.isfile(os.path.join(tt_dir, f))),
        None
    )
    test_path = next(
        (os.path.join(tt_dir, f) for f in ["UNSW_NB15_testing-set.csv", "UNSW-NB15-testing.csv"]
         if os.path.isfile(os.path.join(tt_dir, f))),
        None
    )
    if train_path and test_path:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        df = pd.concat([df_train, df_test], ignore_index=True)
        print(f"✅ Loaded official train/test dataset: {df.shape}")

# Fallback: load 4 split files
if df is None:
    print("⚙️ Loading UNSW-NB15_1..4.csv split files...")
    split_paths = sorted(glob.glob(os.path.join(dataset_dir, "UNSW-NB15_*.csv")))
    if not split_paths:
        raise FileNotFoundError("❌ No UNSW-NB15_*.csv files found.")
    parts = [pd.read_csv(p) for p in split_paths]
    df = pd.concat(parts, ignore_index=True)
    print(f"✅ Loaded combined split files: {df.shape}")

    # Try to attach GT labels
    gt_path = None
    for name in ["NUSW-NB15_GT.csv", "UNSW-NB15_GT.csv"]:
        path = os.path.join(dataset_dir, name)
        if os.path.isfile(path):
            gt_path = path
            break
    if gt_path:
        gt = pd.read_csv(gt_path)
        if len(gt) == len(df):
            for col in ["label", "attack_cat"]:
                if col in gt.columns:
                    df[col] = gt[col].values
            print("✅ Attached ground-truth labels.")
        else:
            print("⚠️ GT rows do not match feature rows. Skipping attachment.")

# -----------------------------
# Step 3: Prepare data
# -----------------------------
y_label = df["label"] if "label" in df.columns else None
y_attack = df["attack_cat"] if "attack_cat" in df.columns else None
X = df.drop(columns=[c for c in ["label", "attack_cat"] if c in df.columns], errors="ignore")

# Drop high-cardinality identifiers
to_drop = [c for c in ["srcip", "dstip", "stime", "ltime", "id"] if c in X.columns]
if to_drop:
    print(f"🧹 Dropping ID-like columns: {to_drop}")
    X = X.drop(columns=to_drop, errors="ignore")

# -----------------------------
# Step 4: Encode categoricals
# -----------------------------
print("🔣 Encoding categorical columns...")
X_enc = pd.get_dummies(X, drop_first=True)
print(f"✅ Encoded shape: {X_enc.shape}")

# -----------------------------
# Step 5: Standardize features
# -----------------------------
print("⚙️ Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_enc)

# -----------------------------
# Step 6: Apply PCA (4 components)
# -----------------------------
print("🎯 Applying PCA (4 components for 4 qubits)...")
pca = PCA(n_components=4, svd_solver="full", random_state=0)
X_pca = pca.fit_transform(X_scaled)
cumvar = pca.explained_variance_ratio_.sum()
print(f"✅ Reduced to 4 components | cumulative variance = {cumvar:.4f}")

# -----------------------------
# Step 7: Save reduced dataset
# -----------------------------
cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=cols)
if y_label is not None:
    df_pca["label"] = y_label.values
if y_attack is not None:
    df_pca["attack_cat"] = y_attack.values

out_path = os.path.join(base_dir, "UNSW_NB15_PCA_4.csv")
df_pca.to_csv(out_path, index=False)
print(f"💾 Saved reduced dataset: {out_path}")

# Save PCA + Scaler artifacts
dump(scaler, os.path.join(base_dir, "unsw_scaler.joblib"))
dump(pca, os.path.join(base_dir, "unsw_pca.joblib"))
print("💾 Saved artifacts: unsw_scaler.joblib, unsw_pca.joblib")

print("\n🎉 Done! Dataset successfully reduced for Q-ADAM (4 qubits).")
