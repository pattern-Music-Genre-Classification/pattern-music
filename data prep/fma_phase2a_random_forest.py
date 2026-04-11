"""
Phase 2A — Random Forest Baseline (Tabular Features)

Train RandomForestClassifier on pre-extracted tabular features (88 dims) from Phase 1B .pt files.
Tabular features are already StandardScaler-scaled — use directly.

Usage:
    python fma_phase2a_random_forest.py
"""

import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
PT_DIR     = BASE_DIR / 'output' / 'pt_features'
MODELS_DIR = BASE_DIR / 'output' / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

GENRE_LABELS = [
    'Electronic', 'Experimental', 'Folk', 'Hip-Hop',
    'Instrumental', 'International', 'Pop', 'Rock',
]
NUM_CLASSES = 8

# 88 tabular feature names (matches extract_tabular_features order in Phase 1A)
FEATURE_NAMES = (
    [f'mfcc_mean_{i}' for i in range(20)] +
    [f'mfcc_std_{i}'  for i in range(20)] +
    ['centroid_mean', 'centroid_std'] +
    ['bandwidth_mean', 'bandwidth_std'] +
    ['rolloff_mean', 'rolloff_std'] +
    [f'chroma_mean_{i}' for i in range(12)] +
    [f'chroma_std_{i}'  for i in range(12)] +
    ['zcr_mean', 'zcr_std'] +
    ['rms_mean', 'rms_std'] +
    [f'contrast_mean_{i}' for i in range(7)] +
    [f'contrast_std_{i}'  for i in range(7)]
)
assert len(FEATURE_NAMES) == 88, f'Expected 88 features, got {len(FEATURE_NAMES)}'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split(split_name: str, pt_dir: Path = PT_DIR):
    """
    Load tabular features + labels from all .pt files in pt_dir/split_name/.
    Returns X (N, 88) float32 numpy array and y (N,) int array.
    """
    files = sorted((pt_dir / split_name).glob('*.pt'))
    if not files:
        raise FileNotFoundError(f'No .pt files found in {pt_dir / split_name}')

    X, y = [], []
    for f in files:
        d = torch.load(f, weights_only=True)
        X.append(d['tabular'].to(torch.float32).numpy())
        y.append(int(d['label']))

    X_arr = np.stack(X)
    y_arr = np.array(y, dtype=np.int64)
    print(f'  [{split_name}] {X_arr.shape[0]:,} samples, {X_arr.shape[1]} features')
    assert not np.isnan(X_arr).any(), f'NaN in {split_name} tabular features!'
    return X_arr, y_arr


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train RandomForestClassifier with sensible defaults."""
    rf = RandomForestClassifier(
        n_estimators     = 500,
        max_depth        = None,   # grow fully
        min_samples_leaf = 2,
        n_jobs           = -1,     # all CPU cores
        random_state     = 42,
        class_weight     = 'balanced',
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    print(f'  Training time: {time.time() - t0:.1f}s')
    return rf


# ---------------------------------------------------------------------------
# Hyperparameter search (optional)
# ---------------------------------------------------------------------------
def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 20,
    cv: int = 3,
) -> dict:
    """
    RandomizedSearchCV over RF hyperparameters.
    Returns best_params dict.
    """
    param_dist = {
        'n_estimators':     [100, 200, 300, 500, 700, 1000],
        'max_depth':        [10, 20, 30, None],
        'min_samples_leaf': [1, 2, 3, 5],
        'max_features':     ['sqrt', 'log2'],
    }
    base_rf = RandomForestClassifier(
        n_jobs=-1, random_state=42, class_weight='balanced'
    )
    search = RandomizedSearchCV(
        base_rf,
        param_distributions = param_dist,
        n_iter              = n_iter,
        cv                  = cv,
        scoring             = 'f1_weighted',
        n_jobs              = -1,
        random_state        = 42,
        verbose             = 1,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f'  Search time: {time.time() - t0:.1f}s')
    print(f'  Best CV F1:  {search.best_score_:.4f}')
    print(f'  Best params: {search.best_params_}')
    return search.best_params_


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    rf: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    save_confusion: Path | None = None,
) -> tuple[float, np.ndarray]:
    """
    Compute weighted F1 + print classification report.
    Optionally save confusion matrix heatmap.
    Returns (weighted_f1, y_pred).
    """
    y_pred = rf.predict(X)
    f1 = f1_score(y, y_pred, average='weighted')
    print(f'\n[{split_name.upper()}] Weighted F1: {f1:.4f}')
    print(classification_report(y, y_pred, target_names=GENRE_LABELS, digits=3))

    if save_confusion is not None:
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot      = True,
            fmt        = 'd',
            cmap       = 'Blues',
            xticklabels = GENRE_LABELS,
            yticklabels = GENRE_LABELS,
            ax         = ax,
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(f'Confusion Matrix — {split_name} (weighted F1={f1:.4f})', fontsize=13)
        plt.tight_layout()
        fig.savefig(save_confusion, dpi=150)
        plt.close(fig)
        print(f'  Saved: {save_confusion}')

    return f1, y_pred


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------
def latency_benchmark(rf: RandomForestClassifier, X: np.ndarray, n_trials: int = 1000) -> dict:
    """
    Measure single-sample inference latency (simulate real-time use).
    Returns dict with latency_ms and rtf.
    """
    sample = X[:1]
    # warm-up
    for _ in range(10):
        rf.predict(sample)

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        rf.predict(sample)
        times.append(time.perf_counter() - t0)

    latency_ms = float(np.mean(times) * 1000)
    rtf = latency_ms / 3000.0   # chunk duration = 3000 ms
    print(f'\nLatency: {latency_ms:.3f} ms (mean over {n_trials} trials)')
    print(f'RTF:     {rtf:.6f}  (target < 1.0)  ✓' if rtf < 1.0 else f'RTF:     {rtf:.6f}  ⚠ > 1.0')
    return {'latency_ms': latency_ms, 'rtf': rtf}


# ---------------------------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------------------------
def plot_feature_importance(
    rf: RandomForestClassifier,
    top_n: int = 20,
    save_path: Path | None = None,
) -> None:
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [FEATURE_NAMES[i] for i in indices]
    top_vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), top_vals[::-1], align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel('Mean decrease in impurity', fontsize=11)
    ax.set_title(f'Random Forest — Top {top_n} Feature Importances', fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'  Saved: {save_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
def save_model(rf: RandomForestClassifier, path: Path = MODELS_DIR / 'rf_baseline.pkl') -> None:
    with open(path, 'wb') as f:
        pickle.dump(rf, f)
    size_mb = path.stat().st_size / 1e6
    print(f'  Saved model: {path}  ({size_mb:.1f} MB)')


def load_model(path: Path = MODELS_DIR / 'rf_baseline.pkl') -> RandomForestClassifier:
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_results(results: dict, path: Path = MODELS_DIR / 'rf_results.json') -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'  Saved results: {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(run_hyperparameter_search: bool = False) -> None:
    print('=' * 60)
    print('  Phase 2A — Random Forest Baseline')
    print('=' * 60)

    # 1. Load data
    print('\nLoading tabular features from .pt files...')
    X_train, y_train = load_split('training')
    X_val,   y_val   = load_split('validation')
    X_test,  y_test  = load_split('test')

    # 2. (Optional) hyperparameter search
    best_params = {}
    if run_hyperparameter_search:
        print('\nRunning hyperparameter search (~5 min)...')
        best_params = hyperparameter_search(X_train, y_train)

    # 3. Train
    print('\nTraining Random Forest...')
    if best_params:
        rf = RandomForestClassifier(
            **best_params,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
        )
        t0 = time.time()
        rf.fit(X_train, y_train)
        print(f'  Training time: {time.time() - t0:.1f}s')
    else:
        rf = train_random_forest(X_train, y_train)

    # 4. Validation evaluation
    print('\nValidation evaluation...')
    val_f1, _ = evaluate(
        rf, X_val, y_val, 'validation',
        save_confusion=MODELS_DIR / 'rf_confusion_val.png',
    )

    # 5. Test evaluation
    print('\nTest evaluation...')
    test_f1, y_test_pred = evaluate(
        rf, X_test, y_test, 'test',
        save_confusion=MODELS_DIR / 'rf_confusion_test.png',
    )

    # 6. Latency
    latency = latency_benchmark(rf, X_test)

    # 7. Feature importance
    print('\nPlotting feature importance...')
    plot_feature_importance(rf, top_n=20, save_path=MODELS_DIR / 'rf_feature_importance.png')

    # 8. Save
    print('\nSaving model and results...')
    save_model(rf)

    results = {
        'val_f1_weighted':  float(val_f1),
        'test_f1_weighted': float(test_f1),
        'per_class_f1': dict(zip(
            GENRE_LABELS,
            f1_score(y_test, y_test_pred, average=None).tolist(),
        )),
        'latency_ms':      latency['latency_ms'],
        'rtf':             latency['rtf'],
        'n_estimators':    rf.n_estimators,
        'n_train_samples': int(X_train.shape[0]),
        'best_params':     best_params,
    }
    save_results(results)

    # 9. Summary
    print('\n' + '=' * 60)
    print('  SUMMARY')
    print('=' * 60)
    print(f'  Val  weighted F1 : {val_f1:.4f}')
    print(f'  Test weighted F1 : {test_f1:.4f}')
    print(f'  Latency          : {latency["latency_ms"]:.3f} ms')
    print(f'  RTF              : {latency["rtf"]:.6f}  (< 1.0 = real-time capable)')
    print(f'\n  Per-class F1 on test:')
    per_class = f1_score(y_test, y_test_pred, average=None)
    for genre, score in zip(GENRE_LABELS, per_class):
        bar = '#' * int(score * 30)
        print(f'    {genre:<15} {score:.3f}  {bar}')
    print('\nDone.')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    main(run_hyperparameter_search=False)
