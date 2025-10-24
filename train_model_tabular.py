#!/usr/bin/env python3
"""
train_model_tabular.py
Author: Yanbing Wang
Date: 2025-10-23

One-click, reproducible training and evaluation for tabular travel-time prediction.

What this script does
- Loads a prepared parquet dataset (e.g., database/i10-broadway/X_full_1h.parquet)
- Applies light preprocessing (log transforms for lag features)
- Chronologically splits data into train/test by time_bin
- Balances the training set by downsampling non-event rows
- Trains several model families (linear, tree-based, optional XGBoost)
 - Evaluates with RMSE/MAE/R2 and cross-validation (default CV=5)
 - Prints and saves per-model top-10 feature importances
- Saves metrics (including feature importances), and trained models to an output directory

Usage (examples)
- Minimal (runs all fast defaults):
    python train_model_tabular.py
- Custom paths and model group (xgb only), with CV=5:
    python train_model_tabular.py --data-path database/i10-broadway/X_full_1h.parquet \
        --output-dir models/tabular_run --model-group xgb --cv 5
- Faster run (reduced models, fewer CV folds):
    python train_model_tabular.py --fast

Notes
- XGBoost models require the xgboost package; if not installed, they will be skipped.
- The dataset is expected to include:
    - target column: 'tt_per_mile' (configurable via --target-col)
    - event column: 'evt_total' (for balancing; configurable via --event-col)
    - time index/column named 'time_bin' for chronological split.
"""

from __future__ import annotations

# ===============
# Imports (top)
# ===============
from pathlib import Path
import argparse
import json
import sys
import time
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# XGBoost availability handled in src.utils

import joblib
try:
    import matplotlib.pyplot as plt  # type: ignore
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

# Import shared helpers
try:
    from src.utils import (
        log,
        ensure_output_dir,
        load_dataset,
        attach_time_index,
        preprocess,
        time_split,
        balance_training,
        get_feature_names,
        extract_linear_feature_importance,
        extract_tree_feature_importance,
        extract_tree_feature_importance,
        make_preprocessor,
        make_model,
        make_rf,
        make_gbrt,
        make_xgb,
        XGB_AVAILABLE,
    )
except Exception:
    # Fallback: ensure project root is on sys.path when run from subfolders
    root_dir = Path(__file__).resolve().parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.utils import (
        log,
        ensure_output_dir,
        load_dataset,
        attach_time_index,
        preprocess,
        time_split,
        balance_training,
        get_feature_names,
        extract_linear_feature_importance,
        extract_tree_feature_importance,
        make_preprocessor,
        make_model,
        make_rf,
        make_gbrt,
        make_xgb,
        XGB_AVAILABLE,
    )



# ===============
# Features & Models
# ===============

def feature_config() -> Dict[str, List[str]]:
    cyc_features = [
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'hour_of_week_sin', 'hour_of_week_cos', 'is_weekend'
    ]
    evt_features = ['evt_cat_unplanned', 'evt_cat_planned']
    lag_features = ['log_lag1_tt_per_mile', 'log_lag2_tt_per_mile', 'log_lag3_tt_per_mile']
    road_features = ['miles', 'reference_speed', 'curve', 'onramp', 'offramp']
    full_features = cyc_features + evt_features + lag_features + road_features
    return {
        "cyc": cyc_features,
        "evt": evt_features,
        "lag": lag_features,
        "road": road_features,
        "full": full_features,
    }


# moved: make_preprocessor, make_model, make_rf, make_gbrt, make_xgb -> src.utils


def build_models(groups: List[str], features: Dict[str, List[str]], transform_target: bool = True, fast: bool = False) -> Dict[str, object]:
    models: Dict[str, object] = {}

    # Linear family
    if 'lr' in groups or 'all' in groups:
        if fast:
            variants = {
                'lr_full': make_model(make_preprocessor(features['full']), LinearRegression(), transform_target)
            }
        else:
            variants = {
                'lr_base':       make_model(make_preprocessor(features['road']), LinearRegression(), transform_target),
                'lr_base_lags':  make_model(make_preprocessor(features['road'] + features['lag']), LinearRegression(), transform_target),
                'lr_full':       make_model(make_preprocessor(features['full']), LinearRegression(), transform_target),
                'lr_cyc':        make_model(make_preprocessor(features['cyc'] + features['road']), LinearRegression(), transform_target),
                'lr_cyc_lags':   make_model(make_preprocessor(features['cyc'] + features['road'] + features['lag']), LinearRegression(), transform_target),
                'lr_evt':        make_model(make_preprocessor(features['evt'] + features['road']), LinearRegression(), transform_target),
                'lr_evt_lags':   make_model(make_preprocessor(features['evt'] + features['road'] + features['lag']), LinearRegression(), transform_target),
                'ridge_full':    make_model(make_preprocessor(features['full']), Ridge(alpha=1.0), transform_target),
                'lasso_full':    make_model(make_preprocessor(features['full']), Lasso(alpha=0.01), transform_target),
            }
        models.update(variants)

    # Tree family
    if 'tree' in groups or 'all' in groups:
        if fast:
            variants = {
                'rf_full':   make_model(make_preprocessor(features['full'], scale=False), make_rf()),
                'gbrt_full': make_model(make_preprocessor(features['full'], scale=False), make_gbrt()),
            }
        else:
            variants = {
                'rf_base':       make_model(make_preprocessor(features['road'], scale=False), make_rf()),
                'rf_base_lags':  make_model(make_preprocessor(features['road'] + features['lag'], scale=False), make_rf()),
                'rf_cyc':        make_model(make_preprocessor(features['cyc'] + features['road'], scale=False), make_rf()),
                'rf_cyc_lags':   make_model(make_preprocessor(features['cyc'] + features['road'] + features['lag'], scale=False), make_rf()),
                'rf_evt':        make_model(make_preprocessor(features['evt'] + features['road'], scale=False), make_rf()),
                'rf_evt_lags':   make_model(make_preprocessor(features['evt'] + features['road'] + features['lag'], scale=False), make_rf()),
                'rf_full':       make_model(make_preprocessor(features['full'], scale=False), make_rf()),
                'gbrt_base':     make_model(make_preprocessor(features['road'], scale=False), make_gbrt()),
                'gbrt_base_lags': make_model(make_preprocessor(features['road'] + features['lag'], scale=False), make_gbrt()),
                'gbrt_cyc':      make_model(make_preprocessor(features['cyc'] + features['road'], scale=False), make_gbrt()),
                'gbrt_cyc_lags': make_model(make_preprocessor(features['cyc'] + features['road'] + features['lag'], scale=False), make_gbrt()),
                'gbrt_evt':      make_model(make_preprocessor(features['evt'] + features['road'], scale=False), make_gbrt()),
                'gbrt_evt_lags': make_model(make_preprocessor(features['evt'] + features['road'] + features['lag'], scale=False), make_gbrt()),
                'gbrt_full':     make_model(make_preprocessor(features['full'], scale=False), make_gbrt()),
            }
        models.update(variants)

    # XGBoost family
    if ('xgb' in groups or 'all' in groups) and XGB_AVAILABLE:
        if fast:
            variants = {
                'xgb_full': make_model(make_preprocessor(features['full'], scale=False), make_xgb())
            }
        else:
            variants = {
                'xgb_base':      make_model(make_preprocessor(features['road'], scale=False), make_xgb()),
                'xgb_base_lags': make_model(make_preprocessor(features['road'] + features['lag'], scale=False), make_xgb()),
                'xgb_cyc':       make_model(make_preprocessor(features['cyc'] + features['road'], scale=False), make_xgb()),
                'xgb_cyc_lags':  make_model(make_preprocessor(features['cyc'] + features['road'] + features['lag'], scale=False), make_xgb()),
                'xgb_evt':       make_model(make_preprocessor(features['evt'] + features['road'], scale=False), make_xgb()),
                'xgb_evt_lags':  make_model(make_preprocessor(features['evt'] + features['road'] + features['lag'], scale=False), make_xgb()),
                'xgb_full':      make_model(make_preprocessor(features['full'], scale=False), make_xgb()),
            }
        models.update(variants)
    elif ('xgb' in groups or 'all' in groups) and not XGB_AVAILABLE:
        log("xgboost not installed; skipping XGB models")

    return models


# ===============
# Evaluation
# ===============

def evaluate_model(name: str, model, X_train: pd.DataFrame, y_train: pd.Series | np.ndarray,
                   X_test: pd.DataFrame, y_test: pd.Series | np.ndarray, cv: int = 0) -> Dict[str, float | str]:
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_pred = model.predict(X_test)
    rmse = float(root_mean_squared_error(y_test, y_pred)) # rmse(y,yhat) of original scale
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    cv_mean = 0.0
    cv_std = 0.0
    if cv and cv > 1:
        scores = np.sqrt(-cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)) # rmse (y,yhat) of original scale
        cv_mean = float(scores.mean())
        cv_std = float(scores.std())

    return {
        'model': name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'fit_time': float(fit_time),
        'cv_rmse_mean': cv_mean,
        'cv_rmse_std': cv_std,
    }


def train_and_evaluate(models: Dict[str, object], X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series, cv: int = 0) -> pd.DataFrame:
    results = []
    for name, model in models.items():
        log(f"Training: {name}")
        metrics = evaluate_model(name, model, X_train, y_train, X_test, y_test, cv=cv)
        results.append(metrics)
    results_df = pd.DataFrame(results)
    sort_col = 'cv_rmse_mean' if cv and 'cv_rmse_mean' in results_df.columns else 'rmse'
    results_df = results_df.sort_values(sort_col).reset_index(drop=True)
    return results_df


def extract_linear_feature_importance(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    try:
        pipe = model.regressor_
        pre = pipe.named_steps['pre']
        reg = pipe.named_steps['reg']
        coefs = getattr(reg, 'coef_', None)
        if coefs is None:
            return None
        # For multi-output or 2D shapes
        coef_arr = np.ravel(coefs)
        return pd.DataFrame({
            'feature': feature_names,
            'coef': coef_arr
        }).sort_values('coef', key=np.abs, ascending=False)
    except Exception:
        return None


def extract_tree_feature_importance(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    try:
        pipe = model.regressor_
        pre = pipe.named_steps['pre']
        reg = pipe.named_steps['reg']
        importances = getattr(reg, 'feature_importances_', None)
        if importances is None:
            return None
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    except Exception:
        return None


def get_feature_names(model) -> Optional[List[str]]:
    """Extract original feature names from the fitted model's preprocessor.
    Returns a list without the ColumnTransformer prefix (e.g., 'num__').
    """
    try:
        pipe = model.regressor_
        pre = pipe.named_steps['pre']
        raw = list(pre.get_feature_names_out())
        # Strip transformer prefix if present, e.g., 'num__feature'
        names = [n.split('__', 1)[1] if '__' in n else n for n in raw]
        return names
    except Exception:
        return None


def save_xgb_full_pie(models: Dict[str, object], feats: Dict[str, List[str]], output_dir: Path) -> None:
    """Compute xgb_full feature importances and visualize as a nested (group + feature) pie using matplotlib.

    - Outer ring: feature groups ('cyc', 'evt', 'lag', 'road', 'other')
    - Inner ring: sub-features, colored using shades from tab20c clusters
    - Saves PNG + CSVs
    """
    if 'xgb_full' not in models:
        log("xgb_full not available in current models; skipping pie chart.")
        return
    if not MPL_AVAILABLE:
        log("matplotlib not available; skipping pie chart.")
        return

    try:
        model = models['xgb_full']
        pipe = model.regressor_  # type: ignore[attr-defined]
        reg = pipe.named_steps['reg']
        importances = getattr(reg, 'feature_importances_', None)
        if importances is None:
            log("No feature_importances_ found on xgb_full; skipping pie chart.")
            return

        feat_names = get_feature_names(model) or feats.get('full', [])
        n = min(len(importances), len(feat_names))
        if n == 0:
            log("Empty features or importances for xgb_full; skipping pie chart.")
            return

        feat_names = feat_names[:n]
        importances = np.asarray(importances[:n], dtype=float)

        # Map features to groups
        group_map = {f: g for g in ['cyc', 'evt', 'lag', 'road'] for f in feats.get(g, [])}
        df = pd.DataFrame({
            'feature': feat_names,
            'importance': importances,
            'group': [group_map.get(f, 'other') for f in feat_names]
        })
        df = df[df['importance'] > 0]
        if df.empty:
            log("All importances are zero; skipping pie chart.")
            return

        # Normalize
        df['importance'] = df['importance'] / df['importance'].sum()

        # Save CSVs
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / 'xgb_full_feature_importance_detailed.csv', index=False)
        df_group = df.groupby('group', as_index=False)['importance'].sum()
        df_group.to_csv(output_dir / 'xgb_full_feature_importance_by_group.csv', index=False)
        log(f"Saved grouped feature importances CSVs to: {output_dir}")

        # Order groups: preferred order then any remaining
        present = df_group['group'].tolist()
        preferred = [g for g in ['cyc', 'evt', 'lag', 'road', 'other'] if g in present]
        remaining = [g for g in present if g not in preferred]
        groups = preferred + remaining

        # Build values for nested pie
        vals = []
        inner_labels = []
        for g in groups:
            sub = df[df['group'] == g].sort_values('importance', ascending=False)
            vals.append(sub['importance'].values)
            inner_labels.extend(sub['feature'].tolist())

        outer_vals = [v.sum() for v in vals]
        inner_vals = np.concatenate(vals).tolist()

        # Colors from tab20c, cluster per group (indices 0,4,8,12,16 as bases)
        cmap = plt.colormaps['tab20c']
        palette = cmap(np.linspace(0, 1, 20))
        base_idx = [0, 4, 8, 12, 16]
        outer_colors = [palette[base_idx[i % len(base_idx)]] for i in range(len(groups))]

        inner_colors = []
        for i, v in enumerate(vals):
            b = base_idx[i % len(base_idx)]
            cluster = [palette[min(b + 1, 19)], palette[min(b + 2, 19)], palette[min(b + 3, 19)]]
            if len(v) <= len(cluster):
                inner_colors.extend(cluster[:len(v)])
            else:
                repeats = (len(v) + len(cluster) - 1) // len(cluster)
                inner_colors.extend((cluster * repeats)[:len(v)])

        # Plot nested donut chart
        fig, ax = plt.subplots(figsize=(8, 8))
        size = 0.3

        # Compute outer ring labels with percentages
        # outer_vals already sum to 1 (normalized importances), so multiply by 100
        outer_labels = [f"{g} ({p*100:.1f}%)" for g, p in zip(groups, outer_vals)]

        # Outer ring (groups)
        ax.pie(
            outer_vals,
            radius=1.0,
            colors=outer_colors,
            labels=outer_labels,
            labeldistance=1.05,
            wedgeprops=dict(width=size, edgecolor='w')
        )

        # Inner ring (features) with conditional labels based on available radius
        inner_radius = 1.0 - size
        show_inner_labels = inner_radius >= 0.65  # add labels only if there's enough space
        labels_arg = inner_labels if show_inner_labels else None
        inner_pie = ax.pie(
            inner_vals,
            radius=inner_radius,
            colors=inner_colors,
            labels=labels_arg,
            labeldistance=0.7 if show_inner_labels else 1.1,
            textprops=dict(fontsize=8, rotation_mode="anchor"),
            wedgeprops=dict(width=size, edgecolor='w')
        )

        # Rotate inner labels so they radiate from the center and remain upright
        if show_inner_labels:
            wedges_inner = inner_pie[0]
            texts_inner = inner_pie[1]
            for w, t in zip(wedges_inner, texts_inner):
                angle = (w.theta2 + w.theta1) / 2.0  # degrees
                # Keep text upright: flip 180° if on the left half (90°..270°)
                rotation = angle if (angle <= 90 or angle >= 270) else angle + 180
                t.set_rotation(rotation)
                t.set_horizontalalignment('center')
                t.set_verticalalignment('center')

        ax.set(aspect="equal", title="xgb_full Feature Importance (Group + Feature)")
        images_dir = Path('images')
        images_dir.mkdir(exist_ok=True)
        png_path = images_dir / 'xgb_full_feature_importance_nested_pie.png'
        plt.savefig(png_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        log(f"Saved nested pie chart PNG: {png_path}")

    except Exception as e:
        log(f"Warning: unable to compute xgb_full nested pie chart: {e}")

# ===============
# Main
# ===============

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular models for travel-time prediction with reproducible pipeline.")
    parser.add_argument('--data-path', type=str, default=str(Path('database/i10-broadway/X_full_1h.parquet')),
                        help='Path to input parquet file.')
    parser.add_argument('--output-dir', type=str, default=str(Path('models/tabular_run')),
                        help='Directory to save models, metrics, and predictions.')
    parser.add_argument('--target-col', type=str, default='tt_per_mile', help='Name of target column.')
    parser.add_argument('--event-col', type=str, default='evt_total', help='Event column used for class balancing.')
    parser.add_argument('--negative-frac', type=float, default=0.01, help='Fraction of non-event rows to keep in training set.')
    parser.add_argument('--test-split', type=float, default=0.2, help='Fraction for test split by time.')
    parser.add_argument('--model-group', type=str, default='all', choices=['lr', 'tree', 'xgb', 'all'],
                        help='Which family of models to train.')
    parser.add_argument('--cv', type=int, default=5, help='Number of CV folds for training metrics (0 to disable).')
    parser.add_argument('--seed', type=int, default=18, help='Random seed for reproducibility.')
    parser.add_argument('--save-models', action='store_true', help='Save fitted models to output-dir.')
    parser.add_argument('--no-save', action='store_true', help='Do not save models even if --save-models is set.')
    parser.add_argument('--fast', action='store_true', help='Use a reduced set of models and lighter CV.')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    # Load
    df = load_dataset(data_path)

    # Ensure time index exists
    try:
        df = attach_time_index(df, time_col='time_bin')
    except KeyError as e:
        log(str(e))
        return 2

    # Preprocess
    df = preprocess(df)

    # Split
    df_train, df_test = time_split(df, test_ratio=args.test_split, time_level='time_bin')

    # Balance
    df_balanced = balance_training(df_train, event_col=args.event_col, negative_frac=args.negative_frac, seed=args.seed)

    # Features
    feats = feature_config()
    full_features = feats['full']
    # Ensure all features exist; if not, warn and drop missing from list
    missing = [c for c in full_features if c not in df.columns]
    if missing:
        log(f"Warning: missing feature columns: {missing}. They will be ignored.")
        feats = {k: [c for c in v if c in df.columns] for k, v in feats.items()}
        full_features = feats['full']

    target_col = args.target_col
    if target_col not in df.columns:
        log(f"Error: target column '{target_col}' not found in data.")
        return 3

    X_train = df_balanced[full_features]
    y_train = df_balanced[target_col]
    X_test = df_test[full_features]
    y_test = df_test[target_col]

    # Models
    groups = [args.model_group]
    models = build_models(groups, feats, transform_target=True, fast=args.fast)

    # CV fallback for fast
    cv = args.cv
    if args.fast and (cv is None or cv < 2):
        cv = 3

    # Train/Eval
    results_df = train_and_evaluate(models, X_train, y_train, X_test, y_test, cv=cv)
    log("Evaluation results:")
    log(results_df.to_string(index=False))

    # Choose best model
    sort_col = 'cv_rmse_mean' if (cv and 'cv_rmse_mean' in results_df.columns and results_df['cv_rmse_mean'].any()) else 'rmse'
    best_row = results_df.sort_values(sort_col).iloc[0]
    best_name = str(best_row['model'])
    best_model = models[best_name]
    log(f"Best model: {best_name} ({sort_col}={best_row[sort_col]:.4f})")

    # Feature importance: compute and print top-10 for each model; include in metrics.json
    feature_importance_map = {}
    for name, model in models.items():
        feat_names = get_feature_names(model)
        if not feat_names:
            # log(f"[FI] Skipping {name}: could not extract feature names")
            continue
        # Try tree-based importances first
        tree_df = extract_tree_feature_importance(model, feat_names)
        if tree_df is not None:
            top10 = tree_df.head(10)
            # log(f"\n[FI] Top 10 features for {name} (tree importance):\n" + top10.to_string(index=False))
            feature_importance_map[name] = {
                'kind': 'tree_importance',
                'top_features': [
                    {'feature': str(row['feature']), 'score': float(row['importance'])}
                    for _, row in top10.iterrows()
                ]
            }
            continue
        # Fall back to linear coefficients (sorted by absolute value)
        lin_df = extract_linear_feature_importance(model, feat_names)
        if lin_df is not None:
            top10 = lin_df.head(10)
            # log(f"\n[FI] Top 10 features for {name} (|coef|):\n" + top10.to_string(index=False))
            feature_importance_map[name] = {
                'kind': 'coef_abs',
                'top_features': [
                    {'feature': str(row['feature']), 'score': float(row['coef'])}
                    for _, row in top10.iterrows()
                ]
            }
            continue
        log(f"[FI] No feature importance available for {name}")

    # Save metrics and summary (embed per-model top-10 feature importances)
    results_csv = output_dir / 'metrics.csv'
    results_json = output_dir / 'metrics.json'
    results_df.to_csv(results_csv, index=False)
    # Enrich JSON records with feature importance
    records = results_df.to_dict(orient='records')
    for rec in records:
        mname = str(rec.get('model'))
        if mname in feature_importance_map:
            rec['top_features_kind'] = feature_importance_map[mname]['kind']
            rec['top_features'] = feature_importance_map[mname]['top_features']
    with open(results_json, 'w') as f:
        json.dump(records, f, indent=2)

    # Save grouped feature importance pie for xgb_full (if available)
    try:
        save_xgb_full_pie(models, feats, output_dir)
    except Exception as e:
        log(f"Warning: failed to generate xgb_full pie chart: {e}")

    # Save models if requested
    if args.save_models and not args.no_save:
        models_dir = output_dir / 'models'
        ensure_output_dir(models_dir)
        # Save each model
        for name, model in models.items():
            p = models_dir / f'{name}.joblib'
            try:
                joblib.dump(model, p)
            except Exception as e:
                log(f"Warning: failed to save model {name}: {e}")
        # Also save a best_model pointer
        with open(output_dir / 'best_model.txt', 'w') as f:
            f.write(best_name + "\n")
        log(f"Saved {len(models)} models to: {models_dir}")

    # Save a trimmed copy of data columns used for training (optional)
    try:
        used_cols = sorted(set(full_features + [target_col]))
        df_out = df[used_cols].copy()
        # Store only as parquet in output for reproducibility of columns
        out_parquet = output_dir / 'training_data_columns.parquet'
        df_out.reset_index(drop=True).to_parquet(out_parquet)
        log(f"Saved data columns snapshot: {out_parquet}")
    except Exception as e:
        log(f"Warning: unable to save data snapshot: {e}")

    log("Mischief Managed.")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(130)
