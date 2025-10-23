#!/usr/bin/env python3
"""
train_model_lstm.py
Author: Yanbing Wang
Date: 2025-10-23
One-click, reproducible training and evaluation for an LSTM model on tabular time-series sequences per TMC.

What this script does
- Loads a prepared parquet dataset (e.g., database/i10-broadway/X_full_1h.parquet)
- Builds sliding-window sequences per TMC using selected feature columns and target column
- Chronologically splits train/test within each TMC (last fraction for test)
- Trains an LSTM with a Keras Normalization layer adapted on training data
- Evaluates with RMSE/MAE on the test set
- Saves the Keras model and arrays/metadata if requested

Usage (examples)
- Minimal (defaults, will save artifacts if --save-models is set):
    python train_model_lstm.py
- Custom features, output dir, and training budget:
    python train_model_lstm.py \
        --data-path database/i10-broadway/X_full_1h.parquet \
        --output-dir models/lstm_run \
        --seq-len 24 --stride 1 --epochs 50 --batch-size 128 \
        --save-models

Notes
- Expects a MultiIndex with levels ('tmc_code','time_bin') or columns with those names to set as index.
- Default features mirror the notebook (time + event + lag + segment features).
- Set seeds for basic reproducibility; TF determinism may depend on your environment.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import sys
import time
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

# TensorFlow / Keras
# TensorFlow/Keras imports are performed lazily inside functions to avoid import errors at module load time.


# ==================
# Utils
# ==================

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def ensure_out(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    try:
        import importlib
        tf = importlib.import_module('tensorflow')
        tf.random.set_seed(seed)
    except Exception:
        # TensorFlow not available; continue with numpy seed only.
        pass


# ==================
# Data & sequences
# ==================

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    log(f"Loading dataset: {path}")
    df = pd.read_parquet(path)
    log(f"Loaded shape: {df.shape}")
    return df


def ensure_index(df: pd.DataFrame, tmc_level: str = 'tmc_code', time_level: str = 'time_bin') -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex) and {tmc_level, time_level}.issubset(df.index.names):
        return df
    # If columns exist, set them as MultiIndex
    if tmc_level in df.columns and time_level in df.columns:
        df = df.set_index([tmc_level, time_level]).sort_index()
        return df
    raise KeyError("Data must have MultiIndex (tmc_code, time_bin) or columns with those names.")


def default_feature_config() -> Dict[str, List[str]]:
    time_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'hour_of_week_sin', 'hour_of_week_cos', 'is_weekend']
    evt_features = ['evt_cat_unplanned', 'evt_cat_planned']
    lag_features = ['lag1_tt_per_mile', 'lag2_tt_per_mile', 'lag3_tt_per_mile']  # raw lags per notebook
    tmc_features = ['miles', 'reference_speed', 'curve', 'onramp', 'offramp']
    full = time_features + evt_features + lag_features + tmc_features
    return {
        'time': time_features,
        'evt': evt_features,
        'lag': lag_features,
        'tmc': tmc_features,
        'full': full,
    }


def sequence_builder(
    X_full: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
    stride: int
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create sliding window sequences across all TMCs.
    Returns (Xs, ys, meta_df) where meta_df has columns ['tmc_code', 'time_bin'] for the label timestamp.
    """
    Xs, ys, meta = [], [], []
    for tmc, grp in X_full.groupby(level='tmc_code'):
        grp = grp.sort_index(level='time_bin')
        data = grp[feature_cols + [target_col]].to_numpy()
        times = grp.index.get_level_values('time_bin').to_numpy()
        if len(data) <= seq_len:
            continue
        for start in range(0, len(data) - seq_len, stride):
            Xs.append(data[start:start + seq_len, :-1])
            ys.append(data[start + seq_len, -1])
            meta.append((tmc, times[start + seq_len]))
    if not Xs:
        raise ValueError("No sequences generated. Check seq_len/stride or data coverage.")
    Xs = np.stack(Xs)
    ys = np.array(ys)
    meta_df = pd.DataFrame(meta, columns=['tmc_code', 'time_bin'])
    return Xs, ys, meta_df


def chronological_split_per_tmc(
    meta: pd.DataFrame,
    Xs: np.ndarray,
    ys: np.ndarray,
    test_frac: float,
    shuffle_train: bool,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    train_idx, test_idx = [], []
    for tmc, grp in meta.groupby('tmc_code', sort=False):
        grp_sorted = grp.sort_values('time_bin')
        n = len(grp_sorted)
        n_test = max(1, int(n * test_frac))
        train_idx.extend(grp_sorted.index[:-n_test].tolist())
        test_idx.extend(grp_sorted.index[-n_test:].tolist())

    X_train, X_test = Xs[train_idx], Xs[test_idx]
    y_train, y_test = ys[train_idx], ys[test_idx]
    meta_train = meta.loc[train_idx].reset_index(drop=True)
    meta_test = meta.loc[test_idx].reset_index(drop=True)

    if shuffle_train:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]
        meta_train = meta_train.iloc[idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, meta_train, meta_test


# ==================
# Model
# ==================

def build_lstm_model(seq_len: int, n_features: int, hidden_units: int, dropout: float, normalizer) -> object:
    import importlib
    keras_api = importlib.import_module('tensorflow.keras')
    keras_models = importlib.import_module('tensorflow.keras.models')
    keras_layers = importlib.import_module('tensorflow.keras.layers')

    # The model will adapt on training data externally before fit
    model = keras_models.Sequential([
        keras_api.Input(shape=(seq_len, n_features)),
        normalizer,
        keras_layers.LSTM(hidden_units, return_sequences=False),
        keras_layers.Dropout(dropout),
        keras_layers.Dense(32, activation='relu'),
        keras_layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ==================
# IO helpers
# ==================

def save_artifacts(
    model,
    history,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    out_dir: Path
) -> None:
    ensure_out(out_dir)
    # 1) model
    model_path = out_dir / 'model.keras'
    model.save(model_path)
    log(f"Saved model to {model_path}")

    # 2) arrays
    npz_path = out_dir / 'results.npz'
    np.savez_compressed(
        npz_path,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        y_pred=y_pred,
    )
    log(f"Saved arrays to {npz_path}")

    # 3) metadata
    meta_train.to_parquet(out_dir / 'meta_train.parquet')
    meta_test.to_parquet(out_dir / 'meta_test.parquet')
    log("Saved metadata parquet files")

    # 4) metrics (include train/val history and best validation loss for comparison script)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    # Extract Keras History contents
    train_loss = []
    val_loss = []
    best_val = None
    best_epoch = None
    epochs_trained = None
    try:
        hist_dict = getattr(history, 'history', {}) or {}
        train_loss = [float(x) for x in hist_dict.get('loss', [])]
        val_loss = [float(x) for x in hist_dict.get('val_loss', [])]
        if val_loss:
            best_val = float(np.min(val_loss))
            best_epoch = int(int(np.argmin(val_loss)) + 1)
        epochs_trained = int(len(train_loss)) if train_loss else None
    except Exception:
        pass

    metrics_payload = {
        'rmse': rmse,
        'mae': mae,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'seq_len': int(X_train.shape[1]),
        'n_features': int(X_train.shape[2]) if X_train.ndim == 3 else None,
        # histories
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val': best_val,
        'best_epoch': best_epoch,
        'epochs_trained': epochs_trained,
        'val_frac': 0.2,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    log("Saved metrics.json (with training history)")


def save_metrics_json_only(
    history,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path
) -> None:
    """Write metrics.json without saving model/arrays. Mirrors metrics content in save_artifacts."""
    ensure_out(out_dir)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    # Extract history
    train_loss = []
    val_loss = []
    best_val = None
    best_epoch = None
    epochs_trained = None
    try:
        hist_dict = getattr(history, 'history', {}) or {}
        train_loss = [float(x) for x in hist_dict.get('loss', [])]
        val_loss = [float(x) for x in hist_dict.get('val_loss', [])]
        if val_loss:
            best_val = float(np.min(val_loss))
            best_epoch = int(int(np.argmin(val_loss)) + 1)
        epochs_trained = int(len(train_loss)) if train_loss else None
    except Exception:
        pass
    payload = {
        'rmse': rmse,
        'mae': mae,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'seq_len': int(X_train.shape[1]),
        'n_features': int(X_train.shape[2]) if X_train.ndim == 3 else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val': best_val,
        'best_epoch': best_epoch,
        'epochs_trained': epochs_trained,
        'val_frac': 0.2,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(payload, f, indent=2)
    log("Saved metrics.json (metrics only)")


# ==================
# Main
# ==================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train an LSTM for travel-time prediction with sequences per TMC.')
    parser.add_argument('--data-path', type=str, default=str(Path('database/i10-broadway/X_full_1h.parquet')),
                        help='Path to input parquet with MultiIndex (tmc_code, time_bin).')
    parser.add_argument('--output-dir', type=str, default=str(Path('models/lstm_run')),
                        help='Directory to save model and outputs.')
    parser.add_argument('--target-col', type=str, default='tt_per_mile', help='Target column name.')
    parser.add_argument('--seq-len', type=int, default=24, help='Sequence length (lookback).')
    parser.add_argument('--stride', type=int, default=1, help='Sliding window stride.')
    parser.add_argument('--test-frac', type=float, default=0.2, help='Test fraction per TMC (chronological).')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
    parser.add_argument('--hidden-units', type=int, default=64, help='LSTM hidden units.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience on val_loss.')
    parser.add_argument('--shuffle-train', action='store_true', help='Shuffle training sequences after split.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--save-models', action='store_true', help='Save Keras model and outputs.')
    parser.add_argument('--feature-cols', type=str, default='',
                        help='Comma-separated feature columns. Defaults to full set used in notebook if empty.')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    set_seeds(args.seed)

    data_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    ensure_out(out_dir)

    # Load
    df = load_dataset(data_path)
    # Ensure index
    try:
        df = ensure_index(df, 'tmc_code', 'time_bin')
    except KeyError as e:
        log(str(e))
        return 2

    # Features
    feat_cfg = default_feature_config()
    if args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(',') if c.strip()]
    else:
        feature_cols = feat_cfg['full']

    missing = [c for c in feature_cols + [args.target_col] if c not in df.columns]
    if missing:
        log(f"Error: missing required columns: {missing}")
        return 3

    # Sequences
    log("Preparing sequences...")
    Xs, ys, meta = sequence_builder(df, feature_cols, args.target_col, seq_len=args.seq_len, stride=args.stride)
    log(f"Sequences: X={Xs.shape}, y={ys.shape}")

    # Split
    log("Chronological split per TMC...")
    X_train, X_test, y_train, y_test, meta_train, meta_test = chronological_split_per_tmc(
        meta, Xs, ys, test_frac=args.test_frac, shuffle_train=args.shuffle_train, seed=args.seed
    )
    log(f"Train: X={X_train.shape}, Test: X={X_test.shape}")

    # Build model strictly following notebook: create normalizer, adapt, then build model with it
    n_features = X_train.shape[2]
    import importlib
    keras_layers = importlib.import_module('tensorflow.keras.layers')
    normalizer = keras_layers.Normalization(axis=-1)
    normalizer.adapt(X_train.reshape(-1, n_features))
    model = build_lstm_model(args.seq_len, n_features, args.hidden_units, args.dropout, normalizer)

    # Validation split (last 20% of training)
    n_val = max(1, int(0.2 * len(X_train)))
    X_val, y_val = X_train[-n_val:], y_train[-n_val:]
    X_train_sub, y_train_sub = X_train[:-n_val], y_train[:-n_val]

    # EarlyStopping via dynamic import to avoid static import errors
    keras_callbacks = importlib.import_module('tensorflow.keras.callbacks')
    callbacks = [keras_callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)]

    log("Training LSTM...")
    history = model.fit(
        X_train_sub, y_train_sub,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    log("Evaluating on test set...")
    y_pred = model.predict(X_test, verbose=0).squeeze()
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    log(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Save
    if args.save_models:
        save_artifacts(model, history, X_train, y_train, X_test, y_test, y_pred, meta_train, meta_test, out_dir)
    else:
        # Still save metrics for comparison script
        save_metrics_json_only(history, X_train, X_test, y_test, y_pred, out_dir)

    log("Done.")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(130)
