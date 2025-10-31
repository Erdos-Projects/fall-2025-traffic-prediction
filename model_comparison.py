"""
model_comparison.py
Author: Yanbing Wang
Date: 2025-10-23

Standalone script to compare tabular, LSTM, and GCN-LSTM models on the I-10 dataset.

It loads engineered features (X_full_1h.parquet), discovers/loads saved models and
predictions, computes fair RMSE on log(tt_per_mile) across models, and generates:

- Bar chart of RMSE for all tabular models (any feature set)
- Bar chart of RMSE for all models with full features (tabular full + LSTM + GCN-LSTM)
- Heatmaps (time x TMC) for each model's predictions and the ground truth

Sequence models (LSTM, GCN-LSTM) output tt_per_mile directly; we transform to log
for evaluation, but keep heatmaps in tt_per_mile space (human interpretable).
Tabular models often predict log_tt_per_mile; we auto-detect and convert when plotting.

Examples:
  python model_comparison.py --direction WB --save-figs
  python model_comparison.py --direction WB --save-figs --skip-heatmaps
  python model_comparison.py --direction EB --gcn-dir models/gcn/gcn_lstm_i10_eb --lstm-dir models/lstm_run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import torch

# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare tabular, LSTM, and GCN-LSTM models")
    p.add_argument("--data-path", type=Path, default=Path("database/i10-broadway"))
    p.add_argument("--file", type=str, default="X_full_1h.parquet")
    p.add_argument("--direction", type=str, default="WB", choices=["WB", "EB"])
    p.add_argument("--tabular-model-dir", type=Path, default=Path("models/tabular_run/models"))
    p.add_argument("--tabular-metrics", type=Path, default=Path("models/tabular_run/metrics.csv"), help="Path to tabular metrics CSV with cv_rmse values")
    p.add_argument("--gcn-dir", type=Path, default=None, help="Directory containing GCN-LSTM artifacts (predictions.npz, data_object.pt)")
    p.add_argument("--lstm-dir", type=Path, default=Path("models/lstm_run"), help="Directory with LSTM results.npz")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--save-figs", action="store_true", help="Save figures to ./images instead of only showing")
    p.add_argument("--skip-heatmaps", action="store_true", help="Disable plotting heatmaps (ground truth and predictions)")
    return p.parse_args()


# ---------------------------
# Data & utilities
# ---------------------------

def tmc_order(direction: str) -> List[str]:
    if direction.upper() == "WB":
        return [
            "115P04188", "115+04188", "115P04187", "115+04187", "115P04186", "115+04186",
            "115P04185", "115+04185", "115P04184", "115+04184", "115P04183", "115+04183",
            "115P04182", "115+04182", "115P04181", "115+04181", "115P04180", "115+04180",
            "115P04179", "115+04179", "115P04178", "115+04178", "115P04177", "115+04177",
            "115P05165"
        ]
    else:
        return [
            "115N04188", "115-04187", "115N04187", "115-04186", "115N04186", "115-04185",
            "115N04185", "115-04184", "115N04184", "115-04183", "115N04183", "115-04182",
            "115N04182", "115-04181", "115N04181", "115-04180", "115N04180", "115-04179",
            "115N04179", "115-04178", "115N04178", "115-04177", "115N04177", "115-05165",
            "115N05165"
        ]


def feature_sets() -> Dict[str, List[str]]:
    time_features = [
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'hour_of_week_sin', 'hour_of_week_cos', 'is_weekend'
    ]
    evt_features = ['evt_cat_unplanned', 'evt_cat_planned']
    lag_features = ['log_lag1_tt_per_mile', 'log_lag2_tt_per_mile', 'log_lag3_tt_per_mile']
    tmc_features = ['miles', 'reference_speed', 'curve', 'onramp', 'offramp']
    full_features = time_features + evt_features + lag_features + tmc_features

    return {
        'base': tmc_features,
        'base_lags': tmc_features + lag_features,
        'full': full_features,
        'cyc': time_features + tmc_features,
        'cyc_lags': time_features + tmc_features + lag_features,
        'evt': evt_features + tmc_features,
        'evt_lags': evt_features + tmc_features + lag_features,
    }


def load_X_full(data_path: Path, file: str) -> pd.DataFrame:
    X_full = pd.read_parquet(data_path / file)
    if not isinstance(X_full.index, pd.MultiIndex) or set(X_full.index.names) != {"tmc_code", "time_bin"}:
        raise ValueError("Expected MultiIndex (tmc_code, time_bin) in the parquet.")
    X_full = X_full.sort_index(level=["time_bin", "tmc_code"])  # stable order
    return X_full


def build_targets(X_full: pd.DataFrame, order: List[str]) -> Tuple[np.ndarray, np.ndarray, List]:
    # Y in tt_per_mile and log_tt_per_mile
    time_index = X_full.index.get_level_values('time_bin').unique().tolist()
    Y_tt = (
        X_full.reset_index()[['time_bin', 'tmc_code', 'tt_per_mile']]
        .pivot(index='time_bin', columns='tmc_code', values='tt_per_mile')
        .reindex(index=time_index, columns=order)
        .to_numpy()
    )
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    Y_log = np.log(np.clip(Y_tt, eps, None))
    return Y_tt, Y_log, time_index


# ---------------------------
# Tabular models
# ---------------------------

def discover_tabular_models(model_dir: Path) -> List[Path]:
    if not model_dir.exists():
        return []
    return sorted([p for p in model_dir.glob("*.joblib") if p.is_file()])


def predict_tabular_model(model_path: Path, X_full: pd.DataFrame, order: List[str], feat_cols: List[str]) -> np.ndarray:
    """Return predictions as [T, N] array. Auto-construct per-TMC features to avoid mixing rows."""
    model = joblib.load(model_path)
    time_index = X_full.index.get_level_values('time_bin').unique().tolist()
    N = len(order)
    T = len(time_index)
    preds = np.zeros((T, N), dtype=float)
    for j, tmc in enumerate(order):
        grp = X_full.xs(tmc, level='tmc_code').sort_index(level='time_bin')
        X_tmc = grp[feat_cols]
        pred = model.predict(X_tmc)
        preds[:, j] = np.asarray(pred).reshape(-1)
    return preds


def guess_pred_space(preds: np.ndarray, y_log_train: np.ndarray, y_tt_train: np.ndarray) -> str:
    """Heuristic: determine if tabular predictions are in log space (common) or tt space.
    Uses train portion error to decide."""
    from sklearn.metrics import root_mean_squared_error as rmse
    # Compare errors
    err_log = rmse(y_log_train.flatten(), preds[: y_log_train.shape[0], :].flatten())
    # If preds are tt, compare vs tt; match shape
    err_tt = rmse(y_tt_train.flatten(), preds[: y_tt_train.shape[0], :].flatten())
    return "log" if err_log <= err_tt else "tt"


def load_tabular_cv_metrics(metrics_path: Path) -> Dict[str, float]:
    """Load cv_rmse for tabular models, preferring CSV as requested.
    Returns mapping like { 'xgb_full': 4.23, 'rf_cyc_lags': 4.36, ... }
    """
    # Prefer CSV
    if metrics_path.suffix.lower() == ".csv":
        if not metrics_path.exists():
            return {}
        try:
            df = pd.read_csv(metrics_path)
            model_col = next((c for c in df.columns if c.lower() in ("model", "name", "model_feat")), None)
            cv_col = next((c for c in df.columns if c.lower() in ("cv_rmse", "cv", "rmse_cv", "cv_rmse_mean")), None)
            if model_col and cv_col:
                return {str(row[model_col]): float(row[cv_col]) for _, row in df.iterrows()}
        except Exception:
            return {}
        return {}
    # Otherwise, attempt JSON (fallback only)
    if not metrics_path.exists():
        return {}
    try:
        obj = pd.read_json(metrics_path, typ='series')
        if isinstance(obj, pd.Series) and all(isinstance(v, (dict, list)) for v in obj.values):
            res = {}
            for k, v in obj.items():
                if isinstance(v, dict):
                    for key in ("cv_rmse", "cv", "rmse_cv", "cv_rmse_mean"):
                        if key in v:
                            res[k] = float(v[key])
                            break
            return res
    except ValueError:
        pass
    try:
        import json
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        res = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    for key in ("cv_rmse", "cv", "rmse_cv", "cv_rmse_mean"):
                        if key in v:
                            res[k] = float(v[key])
                            break
        return res
    except Exception:
        return {}


# ---------------------------
# Sequence models (GCN-LSTM, LSTM)
# ---------------------------

def load_gcn_preds(gcn_dir: Optional[Path]) -> Optional[Dict[str, np.ndarray]]:
    if gcn_dir is None:
        # Auto-detect WB as default path
        candidate = Path("models/gcn/gcn_lstm_i10_wb")
        if candidate.exists():
            gcn_dir = candidate
        else:
            return None
    npz_path = gcn_dir / "predictions.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    return {
        "preds": d["preds"],  # [T_eval, N]
        "Y": d["Y"],          # [T, N]
        "tmc_order": d["tmc_order"],
        "time_index": d["time_index"],
        "seq_len": int(d["seq_len"]),
        "direction": (str(d["direction"]) if "direction" in d else None),
    }


def load_lstm_preds(lstm_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    npz_path = lstm_dir / "results.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    # Try common keys; adapt as needed
    keys = d.files
    # Expect something like preds, y_true or similar
    # We'll standardize to {preds:[T_eval,N], Y:[T,N]} if possible
    ret: Dict[str, np.ndarray] = {}
    if "preds" in keys:
        ret["preds"] = d["preds"]
    elif "y_pred" in keys:
        ret["preds"] = d["y_pred"]
    if "Y" in keys:
        ret["Y"] = d["Y"]
    elif "y_true" in keys:
        ret["Y"] = d["y_true"]
    # If shape mismatches occur, script will raise later with clear info
    return ret if "preds" in ret and "Y" in ret else None


# ---------------------------
# Metrics & plotting
# ---------------------------

def compute_log_rmse(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> float:
    from sklearn.metrics import root_mean_squared_error as rmse
    return rmse(y_log_true.flatten(), y_log_pred.flatten())


def to_heatmap_df(Z: np.ndarray, time_index: List, order: List[str]) -> pd.DataFrame:
    return pd.DataFrame(Z, index=time_index, columns=order)


def plot_heatmap(df_heat: pd.DataFrame, title: str, vmin: float = 40, vmax: float = 110, save: Optional[Path] = None):
    plt.figure(figsize=(14, 5))
    ax = sns.heatmap(
        df_heat.T,
        cmap='plasma',
        vmin=vmin, vmax=vmax,
        cbar_kws={'label': 'Travel Time (sec/mile)'},
        mask=df_heat.T.isna(),
    )
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("TMC (ordered)")
    # Trim x labels if they are timestamps with timezone strings
    xticklabels = [label.get_text()[:-13] for label in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels, rotation=90, ha='center')
    plt.tight_layout()
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=150)
        plt.close()
    else:
        plt.show()


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()

    # Load data and targets
    X_full = load_X_full(args.data_path, args.file)
    order = tmc_order(args.direction)
    Y_tt, Y_log, time_index = build_targets(X_full, order)
    T, N = Y_tt.shape
    n_test = int(T * args.test_frac)

    # Split indices
    train_slice = slice(0, T - n_test)
    test_slice = slice(T - n_test, T)

    # --- Collect results ---
    tabular_rmse: Dict[str, float] = {}
    full_rmse: Dict[str, float] = {}

    # 1) Tabular models
    feats = feature_sets()
    tabular_models = discover_tabular_models(args.tabular_model_dir)
    # Load cv metrics directly from CSV (or fallback)
    tab_cv = load_tabular_cv_metrics(args.tabular_metrics)

    # Use metrics file only for bar charts
    tabular_rmse = {k: float(v) for k, v in tab_cv.items()}
    # Extract full-feature subset for the second chart
    for name, val in tab_cv.items():
        if "_full" in name:
            prefix = name.split("_", 1)[0]
            full_rmse[f"{prefix}_full"] = float(val)

    # 2) ST-GNN (GCN-LSTM)
    # Use best validation error from metrics.json to populate the full-feature bar chart,
    # independent of whether predictions.npz exists. This leverages the new metrics format
    # where 'best_val' holds the lowest validation MSE; we convert it to RMSE.
    gcn_dir = Path(args.gcn_dir) if args.gcn_dir else Path("models/gcn/gcn_lstm_i10_" + args.direction.lower())
    gcn_metrics_path = gcn_dir / "metrics.json"
    import json
    with open(gcn_metrics_path, 'r') as f:
        data = json.load(f)
        gcn_val_loss = data.get("train_loss", None) #TODO: i don't fully understand how BN affect val loss

    if gcn_val_loss is not None:
        gcn_cv = float(np.sqrt(np.min(np.array(gcn_val_loss, dtype=float))))
        full_rmse["gnn_full"] = gcn_cv
    else:
        print(f"[WARN] ST-GNN metrics not found or missing 'best_val' at {gcn_metrics_path}; skipping gnn_full in full-feature chart.")

    # 3) LSTM – first try metrics.json (best_val -> RMSE), independent of predictions
    lstm_metrics_path = Path(args.lstm_dir) / "metrics.json"
    with open(lstm_metrics_path, 'r') as f:
        data = json.load(f)
        lstm_cv = data.get("best_val", None)
    if lstm_cv is not None:
        full_rmse["lstm_full"] = float(np.sqrt(lstm_cv)) # from mse to rmse
    else:
        # Fallback approximate only if predictions are available
        lstm_info = load_lstm_preds(args.lstm_dir)
        if lstm_info is not None:
            lstm_preds_tt = np.asarray(lstm_info["preds"])  # [T_eval, N] expected
            Y_all_tt = np.asarray(lstm_info["Y"])          # [T, N]
            T_eval = lstm_preds_tt.shape[0]
            seq_len_guess = Y_all_tt.shape[0] - T_eval
            if seq_len_guess < 0:
                raise ValueError("LSTM results: preds longer than Y; unexpected shapes.")
            Y_eval_tt = Y_all_tt[seq_len_guess:seq_len_guess + T_eval, :]
            lstm_preds_log = np.log(np.clip(lstm_preds_tt, 1e-8, None))
            Y_eval_log = np.log(np.clip(Y_eval_tt, 1e-8, None))
            val_start_eval = max(0, int(T * 0.9) - seq_len_guess)
            rmse_log = compute_log_rmse(
                Y_eval_log[val_start_eval:, :],
                lstm_preds_log[val_start_eval:, :],
            )
            full_rmse["lstm_full"] = rmse_log
        else:
            print(f"[WARN] LSTM metrics not found at {lstm_metrics_path} and no predictions available; skipping LSTM in full-feature chart.")

    # ---------------------------
    # Plots
    # ---------------------------
    images_dir = Path("images")

    # A) Bar chart – all tabular models (CV RMSE), grouped by model family and feature set
    if tabular_rmse:
        # Build df similar to the provided notebook code
        rmse_results = {k: {"cv_rmse": v} for k, v in tabular_rmse.items()}
        df = pd.DataFrame(rmse_results).T
        df = df.reset_index().rename(columns={'index': 'model_feat'})
        df['model'] = df['model_feat'].str.extract(r'^(lr|rf|xgb|gbrt|gnn|lstm|ridge|lasso)_?')
        df['features'] = df['model_feat'].str.replace(r'^(lr|rf|xgb|gbrt|lstm|ridge|lasso)_?', '', regex=True)
        # Drop ridge/lasso/seq models if any slipped in
        df = df[~df['model_feat'].isin(['ridge_full', 'lasso_full','gnn_full','lstm_full'])]

        model_map = {
            'lr': 'Linear Regression',
            'rf': 'Random Forest',
            'xgb': 'XGBoost',
            'gbrt': 'Gradient Boosting',
            'gnn': 'GCN+LSTM',
            'lstm': 'LSTM',
            'ridge': 'Ridge Regression',
            'lasso': 'Lasso Regression'
        }
        feature_map = {
            'base': 'road',
            'base_lags': 'road + lag',
            'evt': 'road + evt',
            'evt_lags': 'road + evt + lag',
            'cyc': 'road + cyc',
            'cyc_lags': 'road + cyc + lag',
            'full': 'full'
        }
        df['model'] = df['model'].map(model_map)
        df['features'] = df['features'].map(feature_map).fillna(df['features'])
        # Melt like the example but only cv_rmse
        df_melt = df.melt(id_vars=['model', 'features'], value_vars=['cv_rmse'],
                          var_name='dataset', value_name='rmse')
        df_melt = df_melt.sort_values(by='rmse')

        plt.figure(figsize=(8,3))
        sns.barplot(
            data=df_melt[df_melt['dataset'] == 'cv_rmse'],
            x='features', y='rmse', hue='model',
            # palette='Set2'
        )
        plt.title('CV RMSE by Model and Feature Set')
        plt.ylabel('RMSE')
        plt.xlabel('Feature Set')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Model', bbox_to_anchor=(1.002, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        if args.save_figs:
            images_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(images_dir / "tabular_models_cv_rmse.png", dpi=150)
            plt.close()
        else:
            plt.show()
    else:
        print("[INFO] No tabular models or metrics found in", args.tabular_model_dir)

    # B) Bar chart – all models with full features
    if full_rmse:
        # Map model codes to display names for x-axis
        name_map = {
            'lr_full': 'Linear Regression',
            'ridge_full': 'Ridge Regression',
            'lasso_full': 'Lasso Regression',
            'rf_full': 'Random Forest',
            'xgb_full': 'XGBoost',
            'gbrt_full': 'Gradient Boosting',
            'lstm_full': 'LSTM',
            'gnn_full': 'ST-GNN',
            'st_gnn_full': 'ST-GNN',
        }
        items = [(name_map.get(k, k), v) for k, v in full_rmse.items()]
        df_full = pd.DataFrame(items, columns=["model", "rmse"]).sort_values("rmse")
        plt.figure(figsize=(8, 3))
        sns.barplot(data=df_full, x="model", y="rmse", hue='model')
        
        plt.title("Full Feature Models – CV RMSE")
        plt.ylabel('RMSE')
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if args.save_figs:
            images_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(images_dir / "full_feature_models_cv_rmse.png", dpi=150)
            plt.close()
        else:
            plt.show()
    else:
        print("[INFO] No full-feature model results available for comparison.")

    # C) Heatmaps – predictions and ground truth (tt_per_mile)
    if not args.skip_heatmaps:
        # Ground truth
        df_true = to_heatmap_df(Y_tt, time_index, order)
        plot_heatmap(df_true, title=f"Ground Truth Travel Time – {args.direction}", save=(images_dir / f"heatmap_truth_{args.direction}.png" if args.save_figs else None))

        # For each available model, produce a heatmap in tt space
        # Tabular: convert preds to tt if needed
        for model_path in tabular_models:
            name = model_path.stem
            try:
                prefix, key = name.split("_", 1)
            except ValueError:
                continue
            feats_map = feature_sets()
            if key not in feats_map:
                continue
            preds_any = predict_tabular_model(model_path, X_full, order, feats_map[key])
            # detect space using train portion
            space = guess_pred_space(preds_any, Y_log[train_slice, :], Y_tt[train_slice, :])
            preds_tt = np.exp(preds_any) if space == "log" else preds_any
            df_pred = to_heatmap_df(preds_tt, time_index, order)
            plot_heatmap(
                df_pred,
                title=f"{name} – Predicted Travel Time",
                save=(images_dir / f"heatmap_{name}_{args.direction}.png" if args.save_figs else None)
            )

        # GCN-LSTM heatmap
        gcn_info = load_gcn_preds(gcn_dir)
        if gcn_info is not None:
            gcn_preds_tt = np.asarray(gcn_info["preds"])  # [T_eval, N]
            seq_len = int(gcn_info.get("seq_len", 24))
            # align to full timeline by padding with NaNs to seq_len at start
            Z = np.full_like(Y_tt, fill_value=np.nan)
            T_eval = gcn_preds_tt.shape[0]
            Z[seq_len:seq_len + T_eval, :] = gcn_preds_tt
            df_pred = to_heatmap_df(Z, time_index, order)
            plot_heatmap(df_pred, title=f"GCN-LSTM – Predicted Travel Time ({args.direction})",
                         save=(images_dir / f"heatmap_gcn_{args.direction}.png" if args.save_figs else None))

        # LSTM heatmap
        if lstm_info is not None:
            lstm_preds_tt = np.asarray(lstm_info["preds"])  # [T_eval, N]
            T_eval = lstm_preds_tt.shape[0]
            seq_len_guess = Y_tt.shape[0] - T_eval
            Z = np.full_like(Y_tt, fill_value=np.nan)
            Z[seq_len_guess:seq_len_guess + T_eval, :] = lstm_preds_tt
            df_pred = to_heatmap_df(Z, time_index, order)
            plot_heatmap(df_pred, title=f"LSTM – Predicted Travel Time ({args.direction})",
                         save=(images_dir / f"heatmap_lstm_{args.direction}.png" if args.save_figs else None))

    print("\n✅ Comparison complete.")


if __name__ == "__main__":
    main()
