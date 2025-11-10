# =============================================================================
# Q-ADAM_ENHANCED.py
# Quantum Anomaly Detection - Production Version
# Combines efficiency of V2 with comprehensive metrics from V1
# =============================================================================

import matplotlib
matplotlib.use("Agg")

import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import (
    log_loss, precision_recall_curve, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_fscore_support
)

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_algorithms.optimizers import ADAM

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
print("✅ Libraries imported successfully.\n")

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Data
    "DATA_PATH": "/Users/mac/Downloads/Folder/Fall26/QC/Q-ADAM/QUANTUM-NETWORK-ANOMALY-DETECTION/PCA_REDUCTION/UNSW NB15 PCA 4.csv",
    "TEST_SIZE": 0.2,
    "RANDOM_SEED": 42,

    # Circuit Architecture
    "NUM_QUBITS": 4,
    "NUM_LAYERS": 8,

    # Training
    "EPOCHS": 50,
    "BATCH_SIZE": 512,
    "LR": 1e-3,
    "VAL_FRAC": 0.05,
    "INIT_SCALE": 0.03,

    # Checkpointing
    "MA_WINDOW": 5,
    "MIN_DELTA": 0.01,
    "CKPT": "qvae_model.joblib",

    # Preprocessing
    "CLIP_PERCENTILES": (0.5, 99.5),

    # Output
    "FIG_DIR": "figures/",
}

# Create output directory
os.makedirs(CONFIG["FIG_DIR"], exist_ok=True)

# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================
def load_data():
    """Load and preprocess PCA-reduced dataset."""
    logger.info(f"Loading data from {CONFIG['DATA_PATH']}")

    df = pd.read_csv(CONFIG["DATA_PATH"])
    X = df[['PC1', 'PC2', 'PC3', 'PC4']].values
    y = df['label'].astype(int).values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["TEST_SIZE"],
        random_state=CONFIG["RANDOM_SEED"],
        stratify=y
    )

    # Light outlier clipping on training data
    lo = np.percentile(X_train, CONFIG["CLIP_PERCENTILES"][0], axis=0)
    hi = np.percentile(X_train, CONFIG["CLIP_PERCENTILES"][1], axis=0)
    X_train = np.clip(X_train, lo, hi)
    X_test = np.clip(X_test, lo, hi)

    # Statistics
    n_norm, n_anom = np.bincount(y_train)
    logger.info(f"Train: {X_train.shape[0]} samples ({n_norm} normal, {n_anom} anomaly)")
    logger.info(f"Test:  {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def plot_feature_distributions(X, y, dataset_name="Dataset"):
    """Visualize feature distributions for normal vs anomaly classes."""
    normal, anomaly = X[y == 0], X[y == 1]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        ax.hist(normal[:, i], bins=30, alpha=0.5, label="Normal", density=True)
        ax.hist(anomaly[:, i], bins=30, alpha=0.5, label="Anomaly", density=True)
        ax.set_title(f"PC{i+1}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f"{dataset_name} Feature Distributions", fontsize=14, y=1.00)
    plt.tight_layout()
    save_path = os.path.join(CONFIG["FIG_DIR"], f"{dataset_name}_distributions.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved: {save_path}")

# =============================================================================
# 2. QUANTUM CIRCUIT DEFINITION
# =============================================================================
def make_circuit(num_qubits, num_layers):
    """Build variational quantum circuit (ansatz)."""
    qc = QuantumCircuit(num_qubits)
    x = ParameterVector("x", num_qubits)
    theta = ParameterVector("θ", 2 * num_qubits * num_layers)

    # Input encoding
    for i in range(num_qubits):
        qc.ry(x[i], i)

    # Variational layers
    k = 0
    for _ in range(num_layers):
        for q in range(num_qubits):
            qc.rx(theta[k], q); k += 1
            qc.rz(theta[k], q); k += 1
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

    return qc, x, theta


def interpret(bitstring):
    """Interpret measurement: first qubit determines class."""
    return int(format(bitstring, f"0{CONFIG['NUM_QUBITS']}b")[0])


def visualize_circuit(qc):
    """Save circuit diagram."""
    fig = qc.draw(output='mpl', fold=30, style={'backgroundcolor': '#EEEEEE'})
    fig.suptitle(f'Quantum Circuit ({CONFIG["NUM_QUBITS"]} Qubits, {CONFIG["NUM_LAYERS"]} Layers)',
                 fontsize=12, y=0.98)
    save_path = os.path.join(CONFIG["FIG_DIR"], "quantum_circuit.png")
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# =============================================================================
# 3. DATA SPLITTING
# =============================================================================
def stratified_split(X, y, val_frac):
    """Create stratified validation split."""
    n_idx = np.where(y == 0)[0]
    a_idx = np.where(y == 1)[0]
    np.random.seed(CONFIG["RANDOM_SEED"])
    n_idx, a_idx = np.random.permutation(n_idx), np.random.permutation(a_idx)

    n_val = int(val_frac * len(n_idx))
    a_val = int(val_frac * len(a_idx))

    X_val = np.vstack([X[n_idx[:n_val]], X[a_idx[:a_val]]])
    y_val = np.array([0] * n_val + [1] * a_val)

    X_train = np.vstack([X[n_idx[n_val:]], X[a_idx[a_val:]]])
    y_train = np.array([0] * (len(n_idx) - n_val) + [1] * (len(a_idx) - a_val))

    return X_train, y_train, X_val, y_val

# =============================================================================
# 4. TRAINING
# =============================================================================
def train_qvae(clf, X_train, y_train, X_val, y_val):
    """Train Quantum Neural Network with mini-batch ADAM."""
    B = CONFIG["BATCH_SIZE"]
    history, best_ma = [], float("inf")

    logger.info(f"Training for {CONFIG['EPOCHS']} epochs with batch_size={B}")

    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        # Shuffle data each epoch
        Xs, ys = shuffle(X_train, y_train, random_state=np.random.randint(1e9))

        # Mini-batch training
        for i in range(0, len(Xs), B):
            xb, yb = Xs[i:i+B], ys[i:i+B]
            clf.fit(xb, yb)

        # Validation
        probs = np.clip(clf.predict_proba(X_val)[:, 1], 1e-10, 1 - 1e-10)
        val_loss = log_loss(y_val, probs)
        history.append(val_loss)

        # Moving average
        ma = np.mean(history[-CONFIG["MA_WINDOW"]:])

        # Logging
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:03d}/{CONFIG['EPOCHS']} | Loss={val_loss:.4f} | MA={ma:.4f}")

        # Checkpoint best model
        if ma < best_ma - CONFIG["MIN_DELTA"]:
            best_ma = ma
            joblib.dump(clf, CONFIG["CKPT"])
            logger.info(f"  ✓ Checkpoint saved (MA={best_ma:.4f})")

    # Load best model
    best_model = joblib.load(CONFIG["CKPT"])
    logger.info(f"✅ Training complete. Best MA loss: {best_ma:.4f}\n")

    return best_model, history, best_ma

# =============================================================================
# 5. EVALUATION & VISUALIZATION
# =============================================================================
def evaluate_model(model, X_val, y_val, X_test, y_test):
    """Comprehensive model evaluation with metrics and plots."""

    # --- Find optimal threshold on validation set ---
    probs_val = model.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, probs_val)
    f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = np.argmax(f1)
    threshold = thr[best_idx]

    logger.info(f"Optimal threshold: {threshold:.3f} (Val F1={f1[best_idx]:.3f})")

    # --- Test set predictions ---
    probs_test = model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= threshold).astype(int)

    # --- Metrics ---
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_test, preds_test)
    print(cm)

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, preds_test, target_names=["Normal", "Anomaly"]))

    # --- Confusion Matrix Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'], ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix (Test Set)')
    save_path = os.path.join(CONFIG["FIG_DIR"], "confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved: {save_path}")

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, probs_test)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f'Q-VAE (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    save_path = os.path.join(CONFIG["FIG_DIR"], "roc_curve.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved: {save_path}")

    # --- Probability Distributions ---
    normal_probs = probs_test[y_test == 0]
    anomaly_probs = probs_test[y_test == 1]

    plt.figure(figsize=(8, 5))
    plt.hist(normal_probs, bins=40, alpha=0.6, label='Normal', density=True)
    plt.hist(anomaly_probs, bins=40, alpha=0.6, label='Anomaly', density=True)
    plt.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')
    plt.xlabel('Predicted Anomaly Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Anomaly Score Distribution (Test Set)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    save_path = os.path.join(CONFIG["FIG_DIR"], "probability_distribution.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved: {save_path}")

    return threshold, probs_test, preds_test


def plot_loss_curve(history):
    """Plot training loss curve."""
    epochs = np.arange(1, len(history) + 1)
    ma = pd.Series(history).rolling(CONFIG["MA_WINDOW"], min_periods=1).mean()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history, 'o-', markersize=4, label='Validation Loss', alpha=0.7)
    plt.plot(epochs, ma, 'r-', linewidth=2, label=f'{CONFIG["MA_WINDOW"]}-Epoch MA')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Log-Loss', fontsize=12)
    plt.title('Q-VAE Training Progress', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    save_path = os.path.join(CONFIG["FIG_DIR"], "loss_curve.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved: {save_path}")

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
def main():
    logger.info("="*60)
    logger.info("QUANTUM ANOMALY DETECTION - TRAINING PIPELINE")
    logger.info("="*60 + "\n")

    # Load data
    X_train, X_test, y_train, y_test = load_data()
    plot_feature_distributions(X_train, y_train, "Train")
    plot_feature_distributions(X_test, y_test, "Test")

    # Build quantum circuit
    logger.info("\nBuilding quantum circuit...")
    qc, x, theta = make_circuit(CONFIG["NUM_QUBITS"], CONFIG["NUM_LAYERS"])
    visualize_circuit(qc)
    logger.info(f"Circuit: {CONFIG['NUM_QUBITS']} qubits, {CONFIG['NUM_LAYERS']} layers, {len(theta)} parameters\n")

    # Initialize QNN
    sampler = StatevectorSampler()
    qnn = SamplerQNN(
        sampler=sampler,
        circuit=qc,
        input_params=x,
        weight_params=theta,
        interpret=interpret,
        output_shape=2,
    )

    optimizer = ADAM(maxiter=1, lr=CONFIG["LR"])
    init_params = np.random.uniform(-CONFIG["INIT_SCALE"], CONFIG["INIT_SCALE"], size=len(theta))
    clf = NeuralNetworkClassifier(
        qnn, optimizer=optimizer, initial_point=init_params, warm_start=True
    )

    # Validation split
    X_trn, y_trn, X_val, y_val = stratified_split(X_train, y_train, CONFIG["VAL_FRAC"])
    logger.info(f"Validation: {len(y_val)} samples | Training: {len(y_trn)} samples\n")

    # Train
    model, history, best_ma = train_qvae(clf, X_trn, y_trn, X_val, y_val)
    plot_loss_curve(history)

    # Evaluate
    logger.info("Evaluating on test set...")
    threshold, probs, preds = evaluate_model(model, X_val, y_val, X_test, y_test)

    logger.info("\n" + "="*60)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info(f"Model saved: {CONFIG['CKPT']}")
    logger.info(f"Figures saved: {CONFIG['FIG_DIR']}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
