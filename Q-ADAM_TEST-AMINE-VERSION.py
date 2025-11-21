# =============================================================================
# Q-ADAM_OPTIMIZED.py
# Quantum Anomaly Detection - Auto-Optimized Architecture
# Automatically finds optimal qubits and layers for best performance
# =============================================================================

import matplotlib
matplotlib.use("Agg")

import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import (
    log_loss, precision_recall_curve, confusion_matrix,
    classification_report, roc_curve, auc, f1_score
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
    "DATA_PATH": "PCA_REDUCTION/UNSW NB15 PCA 4.csv",
    "TEST_SIZE": 0.2,
    "RANDOM_SEED": 42,

    # Hyperparameter Search Space
    "QUBIT_RANGE": [2, 3, 4],  # Qubits to test
    "LAYER_RANGE": [2, 4, 6, 8, 10],  # Layers to test
    "SEARCH_EPOCHS": 20,  # Quick epochs for search
    "SEARCH_CV_FOLDS": 3,  # Cross-validation folds
    
    # Final Training (after optimization)
    "FINAL_EPOCHS": 50,
    "BATCH_SIZE": 512,
    "LR": 1e-3,
    "VAL_FRAC": 0.05,
    "INIT_SCALE": 0.03,

    # Checkpointing
    "MA_WINDOW": 5,
    "MIN_DELTA": 0.01,
    "BEST_MODEL_PATH": "qvae_optimized_best.joblib",
    "SEARCH_RESULTS_PATH": "hyperparameter_search_results.csv",

    # Preprocessing
    "CLIP_PERCENTILES": (0.5, 99.5),

    # Output
    "FIG_DIR": "figures_optimized/",
}

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["TEST_SIZE"], 
        random_state=CONFIG["RANDOM_SEED"], stratify=y
    )

    lo = np.percentile(X_train, CONFIG["CLIP_PERCENTILES"][0], axis=0)
    hi = np.percentile(X_train, CONFIG["CLIP_PERCENTILES"][1], axis=0)
    X_train = np.clip(X_train, lo, hi)
    X_test = np.clip(X_test, lo, hi)

    n_norm, n_anom = np.bincount(y_train)
    logger.info(f"Train: {X_train.shape[0]} samples ({n_norm} normal, {n_anom} anomaly)")
    logger.info(f"Test:  {X_test.shape[0]} samples\n")

    return X_train, X_test, y_train, y_test

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

def interpret(bitstring, num_qubits):
    """Interpret measurement: first qubit determines class."""
    return int(format(bitstring, f"0{num_qubits}b")[0])

# =============================================================================
# 3. HYPERPARAMETER SEARCH
# =============================================================================
def quick_train_eval(clf, X_train, y_train, X_val, y_val, epochs):
    """Quick training for hyperparameter search with live logging."""
    B = CONFIG["BATCH_SIZE"]

    total_batches = int(np.ceil(len(X_train) / B))

    for epoch in range(epochs):
        logger.info(f"    ▶ Epoch {epoch+1}/{epochs} started...")
        Xs, ys = shuffle(X_train, y_train, random_state=epoch)

        for batch_idx in range(0, len(Xs), B):
            xb, yb = Xs[batch_idx:batch_idx+B], ys[batch_idx:batch_idx+B]
            clf.fit(xb, yb)

            current_batch = (batch_idx // B) + 1

            # Log every 20 batches + last batch
            if current_batch % 20 == 0 or current_batch == total_batches:
                logger.info(f"       Batch {current_batch}/{total_batches} processed")

    # --- Validation ---
    probs = np.clip(clf.predict_proba(X_val)[:, 1], 1e-10, 1 - 1e-10)
    val_loss = log_loss(y_val, probs)

    prec, rec, thr = precision_recall_curve(y_val, probs)
    f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    best_f1 = np.max(f1_scores)

    logger.info(f"    ✅ Validation completed | Loss: {val_loss:.4f} | F1: {best_f1:.4f}")

    return val_loss, best_f1


def cross_validate_architecture(num_qubits, num_layers, X, y):
    """Cross-validate a specific architecture."""
    skf = StratifiedKFold(n_splits=CONFIG["SEARCH_CV_FOLDS"], 
                          shuffle=True, random_state=CONFIG["RANDOM_SEED"])
    
    cv_losses, cv_f1s = [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # --- FIX: Match feature count to qubit count ---
        X_train = X_train[:, :num_qubits]
        X_val = X_val[:, :num_qubits]

        
        # Build circuit
        qc, x, theta = make_circuit(num_qubits, num_layers)
        
        # Initialize QNN
        sampler = StatevectorSampler()
        qnn = SamplerQNN(
            sampler=sampler,
            circuit=qc,
            input_params=x,
            weight_params=theta,
            interpret=lambda b: interpret(b, num_qubits),
            output_shape=2,
        )
        
        optimizer = ADAM(maxiter=1, lr=CONFIG["LR"])
        init_params = np.random.uniform(-CONFIG["INIT_SCALE"], CONFIG["INIT_SCALE"], 
                                       size=len(theta))
        clf = NeuralNetworkClassifier(
            qnn, optimizer=optimizer, initial_point=init_params, warm_start=True
        )
        
        # Train and evaluate
        val_loss, val_f1 = quick_train_eval(
            clf, X_train, y_train, X_val, y_val, CONFIG["SEARCH_EPOCHS"]
        )
        
        cv_losses.append(val_loss)
        cv_f1s.append(val_f1)
        
        logger.info(f"  Fold {fold+1}/{CONFIG['SEARCH_CV_FOLDS']}: Loss={val_loss:.4f}, F1={val_f1:.4f}")
    
    return np.mean(cv_losses), np.std(cv_losses), np.mean(cv_f1s), np.std(cv_f1s)

def hyperparameter_search(X_train, y_train):
    """Grid search over qubits and layers."""
    logger.info("="*60)
    logger.info("STARTING HYPERPARAMETER SEARCH")
    logger.info("="*60 + "\n")
    
    results = []
    search_space = list(product(CONFIG["QUBIT_RANGE"], CONFIG["LAYER_RANGE"]))
    total = len(search_space)
    
    for idx, (n_qubits, n_layers) in enumerate(search_space, 1):
        n_params = 2 * n_qubits * n_layers
        logger.info(f"\n[{idx}/{total}] Testing: {n_qubits} qubits, {n_layers} layers ({n_params} params)")
        
        start_time = time.time()
        mean_loss, std_loss, mean_f1, std_f1 = cross_validate_architecture(
            n_qubits, n_layers, X_train, y_train
        )
        duration = time.time() - start_time
        
        results.append({
            'qubits': n_qubits,
            'layers': n_layers,
            'params': n_params,
            'mean_loss': mean_loss,
            'std_loss': std_loss,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'duration_sec': duration
        })
        
        logger.info(f"  ✓ Avg Loss: {mean_loss:.4f}±{std_loss:.4f}")
        logger.info(f"  ✓ Avg F1: {mean_f1:.4f}±{std_f1:.4f}")
        logger.info(f"  ✓ Time: {duration:.1f}s")
    
    # Convert to DataFrame and save
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('mean_f1', ascending=False)
    df_results.to_csv(CONFIG["SEARCH_RESULTS_PATH"], index=False)
    
    # Find best configuration
    best = df_results.iloc[0]
    logger.info("\n" + "="*60)
    logger.info("SEARCH COMPLETE - BEST CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Qubits: {int(best['qubits'])}")
    logger.info(f"Layers: {int(best['layers'])}")
    logger.info(f"Parameters: {int(best['params'])}")
    logger.info(f"Mean F1: {best['mean_f1']:.4f}±{best['std_f1']:.4f}")
    logger.info(f"Mean Loss: {best['mean_loss']:.4f}±{best['std_loss']:.4f}")
    logger.info("="*60 + "\n")
    
    return int(best['qubits']), int(best['layers']), df_results

def plot_search_results(df_results):
    """Visualize hyperparameter search results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. F1 Score Heatmap
    pivot_f1 = df_results.pivot(index='layers', columns='qubits', values='mean_f1')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0, 0], cbar_kws={'label': 'F1 Score'})
    axes[0, 0].set_title('Mean F1 Score by Architecture', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Qubits')
    axes[0, 0].set_ylabel('Number of Layers')
    
    # 2. Loss Heatmap
    pivot_loss = df_results.pivot(index='layers', columns='qubits', values='mean_loss')
    sns.heatmap(pivot_loss, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[0, 1], cbar_kws={'label': 'Loss'})
    axes[0, 1].set_title('Mean Loss by Architecture', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Qubits')
    axes[0, 1].set_ylabel('Number of Layers')
    
    # 3. F1 vs Parameters
    axes[1, 0].scatter(df_results['params'], df_results['mean_f1'], 
                       s=100, alpha=0.6, c=df_results['qubits'], cmap='viridis')
    axes[1, 0].errorbar(df_results['params'], df_results['mean_f1'], 
                        yerr=df_results['std_f1'], fmt='none', ecolor='gray', alpha=0.3)
    axes[1, 0].set_xlabel('Number of Parameters', fontsize=11)
    axes[1, 0].set_ylabel('Mean F1 Score', fontsize=11)
    axes[1, 0].set_title('F1 Score vs Model Complexity', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Training Duration
    axes[1, 1].bar(range(len(df_results)), df_results['duration_sec'], 
                   color='steelblue', alpha=0.7)
    axes[1, 1].set_xlabel('Configuration Index', fontsize=11)
    axes[1, 1].set_ylabel('Training Duration (seconds)', fontsize=11)
    axes[1, 1].set_title('Training Time per Configuration', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(CONFIG["FIG_DIR"], "hyperparameter_search.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved: {save_path}")

# =============================================================================
# 4. FINAL TRAINING WITH OPTIMAL ARCHITECTURE
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

def train_final_model(num_qubits, num_layers, X_train, y_train, X_val, y_val):
    """Train final model with optimal architecture."""
    logger.info("\n" + "="*60)
    logger.info(f"TRAINING FINAL MODEL: {num_qubits} qubits, {num_layers} layers")
    logger.info("="*60 + "\n")
    
    # Build circuit
    qc, x, theta = make_circuit(num_qubits, num_layers)
    
    # Initialize QNN
    sampler = StatevectorSampler()
    qnn = SamplerQNN(
        sampler=sampler,
        circuit=qc,
        input_params=x,
        weight_params=theta,
        interpret=lambda b: interpret(b, num_qubits),
        output_shape=2,
    )
    
    optimizer = ADAM(maxiter=1, lr=CONFIG["LR"])
    init_params = np.random.uniform(-CONFIG["INIT_SCALE"], CONFIG["INIT_SCALE"], 
                                   size=len(theta))
    clf = NeuralNetworkClassifier(
        qnn, optimizer=optimizer, initial_point=init_params, warm_start=True
    )
    
    # Training loop
    B = CONFIG["BATCH_SIZE"]
    history, best_ma = [], float("inf")
    
    for epoch in range(1, CONFIG["FINAL_EPOCHS"] + 1):
        Xs, ys = shuffle(X_train, y_train, random_state=epoch)
        
        for i in range(0, len(Xs), B):
            xb, yb = Xs[i:i+B], ys[i:i+B]
            clf.fit(xb, yb)
        
        probs = np.clip(clf.predict_proba(X_val)[:, 1], 1e-10, 1 - 1e-10)
        val_loss = log_loss(y_val, probs)
        history.append(val_loss)
        
        ma = np.mean(history[-CONFIG["MA_WINDOW"]:])
        
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:03d}/{CONFIG['FINAL_EPOCHS']} | Loss={val_loss:.4f} | MA={ma:.4f}")
        
        if ma < best_ma - CONFIG["MIN_DELTA"]:
            best_ma = ma
            joblib.dump(clf, CONFIG["BEST_MODEL_PATH"])
            logger.info(f"  ✓ Checkpoint saved (MA={best_ma:.4f})")
    
    best_model = joblib.load(CONFIG["BEST_MODEL_PATH"])
    logger.info(f"\n✅ Training complete. Best MA loss: {best_ma:.4f}\n")
    
    return best_model, history

# =============================================================================
# 5. EVALUATION
# =============================================================================
def evaluate_model(model, X_val, y_val, X_test, y_test):
    """Comprehensive model evaluation."""
    probs_val = model.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, probs_val)
    f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = np.argmax(f1)
    threshold = thr[best_idx]

    probs_test = model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= threshold).astype(int)

    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_test, preds_test)
    print(cm)

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, preds_test, target_names=["Normal", "Anomaly"]))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs_test)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f'Optimized Q-VAE (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve (Optimized Model)', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    save_path = os.path.join(CONFIG["FIG_DIR"], "roc_curve_optimized.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return threshold, roc_auc

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
def main():
    logger.info("="*60)
    logger.info("QUANTUM ANOMALY DETECTION - AUTO-OPTIMIZED")
    logger.info("="*60 + "\n")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Hyperparameter search
    best_qubits, best_layers, search_results = hyperparameter_search(X_train, y_train)
    plot_search_results(search_results)
    
    # Train final model with optimal architecture
    X_trn, y_trn, X_val, y_val = stratified_split(X_train, y_train, CONFIG["VAL_FRAC"])
    final_model, history = train_final_model(best_qubits, best_layers, X_trn, y_trn, X_val, y_val)
    
    # Evaluate
    threshold, roc_auc = evaluate_model(final_model, X_val, y_val, X_test, y_test)
    
    logger.info("\n" + "="*60)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info(f"Optimal Architecture: {best_qubits} qubits, {best_layers} layers")
    logger.info(f"Test AUC: {roc_auc:.4f}")
    logger.info(f"Model saved: {CONFIG['BEST_MODEL_PATH']}")
    logger.info("="*60)

if __name__ == "__main__":
    main()