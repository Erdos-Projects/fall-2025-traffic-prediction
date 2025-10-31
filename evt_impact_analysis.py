"""
evt_impact_analysis.py
Author: Yanbing Wang
Date: 2025-10-23

A counterfactual analysis script to estimate the impact of planned events on travel time per mile along the I-10 Broadway Curve corridor.
Trains separate models for no-event and planned-event scenarios, predicts travel times, and analyzes the differences.   

To answer questions that agencies may care:
1. What's the best time to do road construction work such that construction-related delay can be minimized?
2. How would improving unplanned event reporting improve travel-time reliability?

Insights we hope to obtain:
1. Time-of-day patterns dominate short-term travel time — meaning proactive congestion management (e.g., signal timing, ramp metering) may be more valuable than reactive incident data.
2. Unplanned event underreporting limits predictive power — improving incident data collection could yield measurable gains in travel-time reliability.
3. Planned work zones can be optimized — data-driven scheduling could reduce construction-related delay and improve traveler satisfaction.

## Workflow
1. train `no_evt` regression model with no evt features and no evt data. This will be a baseline model that predicts "what is likely to be the travel time if there's no planned or unplanned events?"
2. train `plnd_evt` with planned evt data and planned evt feature
3. diff= `plnd_evt.pred`-`no_evt.pred` is model's estimated event impact (note: only as good as model fit; not guaranteed causal)
4. residual = `data`-`plnd_evt.pred` captures the remaining unexplaiend delays due to unplanned events and other unmodeled factors (e.g., weather, #lanes)
5. analysis: average diff by hour of day, day of week; correlation of diff with event rate (spatially and temporally); lagged correlation analysis
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Reuse shared helpers
try:
    from src.utils import (
        log,
        ensure_output_dir,
        load_dataset,
        attach_time_index,
        preprocess,
        make_preprocessor,
        make_model,
        make_xgb,
        make_gbrt,
        XGB_AVAILABLE,
    )
except Exception:
    root_dir = Path(__file__).resolve().parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.utils import (
        log,
        ensure_output_dir,
        load_dataset,
        attach_time_index,
        preprocess,
        make_preprocessor,
        make_model,
        make_xgb,
        make_gbrt,
        XGB_AVAILABLE,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Event impact analysis: trains simple models and outputs figures/metrics.")
    p.add_argument('--data-path', type=str, default=str(Path('database/i10-broadway/X_full_1h.parquet')),
                  help='Path to X_full_1h.parquet')
    p.add_argument('--dir', type=str, default='WB', choices=['WB','EB'],
                  help='Corridor direction for TMC ordering (WB or EB)')
    p.add_argument('--images-dir', type=str, default=str(Path('images')),
                  help='Directory to save output images')
    p.add_argument('--no-event-frac', type=float, default=0.05,
                  help='Fraction of no-event samples to keep for balancing')
    p.add_argument('--buffer', type=int, default=3,
                  help='Buffer window (hours) before/after events for labeling training samples')
    return p.parse_args(argv)


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    import time as _time
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    t0 = _time.time()
    model.fit(X_train, y_train)
    fit_time = _time.time() - t0
    y_pred = model.predict(X_test)
    rmse = float(root_mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return {
        'model': name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'fit_time': fit_time,
    }, y_pred


def savefig(fig, images_dir: Path, name: str):
    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved figure: {path}")


def main(argv=None):
    args = parse_args(argv)
    images_dir = Path(args.images_dir)
    ensure_output_dir(images_dir)

    # 1) Load data and ensure MultiIndex with time_bin
    df = load_dataset(Path(args.data_path))
    try:
        df = attach_time_index(df, time_col='time_bin')
    except KeyError as e:
        log(f"Error: {e}")
        return 2

    # 1b) Ensure log lag features exist
    df = preprocess(df)
    log(f"Columns available: {len(df.columns)}")

    # 2) Train/test split chronologically (80/20)
    time_bins = df.index.get_level_values('time_bin').unique().sort_values()
    split_idx = int(len(time_bins) * 0.8)
    train_times = time_bins[:split_idx]
    test_times = time_bins[split_idx:]
    df_train = df.loc[pd.IndexSlice[:, train_times], :]
    df_test = df.loc[pd.IndexSlice[:, test_times], :]
    log(f"Train size: {len(df_train):,}, Test size: {len(df_test):,}")

    # 3) Label event/no-event samples with ±buffer hours per TMC
    buffer_window = int(args.buffer)
    event_mask = pd.Series(False, index=df_train.index)
    plnd_mask = pd.Series(False, index=df_train.index)
    for tmc, group in df_train.groupby(level='tmc_code'):
        evt_idx = group.index[group['evt_total'] > 0]
        plnd_evt_idx = group.index[group['evt_cat_planned'] > 0]
        if not evt_idx.empty:
            evt_pos = group.index.get_indexer(evt_idx)
            buffer_pos = np.unique(np.concatenate([np.arange(i - buffer_window, i + buffer_window + 1) for i in evt_pos]))
            buffer_pos = buffer_pos[(buffer_pos >= 0) & (buffer_pos < len(group))]
            event_mask.loc[group.index[buffer_pos]] = True
        if not plnd_evt_idx.empty:
            plnd_pos = group.index.get_indexer(plnd_evt_idx)
            buffer_pos = np.unique(np.concatenate([np.arange(i - buffer_window, i + buffer_window + 1) for i in plnd_pos]))
            buffer_pos = buffer_pos[(buffer_pos >= 0) & (buffer_pos < len(group))]
            plnd_mask.loc[group.index[buffer_pos]] = True

    # Build subsets
    df_no_events = df_train[~event_mask].copy()
    df_plnd_events = df_train[plnd_mask].copy()
    # Downsample no-events for balance
    frac = float(args.no_event_frac)
    if 0 < frac < 1:
        df_no_events = df_no_events.sample(frac=frac, random_state=2)
    # Combine for balanced dataset and shuffle
    df_balanced = pd.concat([df_no_events, df_plnd_events]).sample(frac=1, random_state=2).reset_index(drop=True)

    # Log dataset composition
    pct_planned_in_bal = 100 * len(df_plnd_events) / max(len(df_balanced), 1)
    pct_planned_in_train = 100 * len(df_plnd_events) / max(len(df_train), 1)
    ratio_planned_to_no = 100 * len(df_plnd_events) / max(len(df_no_events), 1)
    log(f"Planned events in balanced train: {len(df_plnd_events)}/{len(df_balanced)} = {pct_planned_in_bal:.2f}%")
    log(f"Planned events originally in train: {len(df_plnd_events)}/{len(df_train)} = {pct_planned_in_train:.2f}%")
    log(f"Planned : no-events ratio (balanced set): {len(df_plnd_events)}/{len(df_no_events)} = {ratio_planned_to_no:.2f}%")
    log(f"{len(df_no_events)} : {len(df_balanced)} no_events : balanced samples")

    # 4) Feature sets
    TARGET_COL = 'tt_per_mile'
    time_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'hour_of_week_sin', 'hour_of_week_cos', 'is_weekend']
    tmc_features = ['miles', 'reference_speed', 'curve','onramp', 'offramp']
    no_evt_features = time_features + tmc_features
    plnd_evt_features = time_features + ['evt_cat_planned'] + tmc_features

    # 5) Train simple models (XGB if available, else GBRT)
    reg_no_evt = make_xgb() if XGB_AVAILABLE else make_gbrt()
    reg_planned = make_xgb() if XGB_AVAILABLE else make_gbrt()
    model_no_evt = make_model(make_preprocessor(no_evt_features, scale=False), reg_no_evt)
    model_planned = make_model(make_preprocessor(plnd_evt_features, scale=False), reg_planned)

    X_train_no_evt = df_no_events[no_evt_features]
    y_train_no_evt = df_no_events[TARGET_COL]
    X_test_no_evt = df_test[no_evt_features]
    y_test_no_evt = df_test[TARGET_COL]

    X_train_plnd = df_balanced[plnd_evt_features]
    y_train_plnd = df_balanced[TARGET_COL]
    X_test_plnd = df_test[plnd_evt_features]
    y_test_plnd = df_test[TARGET_COL]

    metrics_no_evt, yhat_no_evt_test = evaluate_model('no_evt', model_no_evt, X_train_no_evt, y_train_no_evt, X_test_no_evt, y_test_no_evt)
    metrics_plnd, yhat_plnd_test = evaluate_model('plnd_evt', model_planned, X_train_plnd, y_train_plnd, X_test_plnd, y_test_plnd)
    log(f"Metrics (no_evt): RMSE={metrics_no_evt['rmse']:.3f}, MAE={metrics_no_evt['mae']:.3f}, R2={metrics_no_evt['r2']:.3f}, fit_time={metrics_no_evt['fit_time']:.2f}s")
    log(f"Metrics (plnd_evt): RMSE={metrics_plnd['rmse']:.3f}, MAE={metrics_plnd['mae']:.3f}, R2={metrics_plnd['r2']:.3f}, fit_time={metrics_plnd['fit_time']:.2f}s")

    # 6) Predict across full dataset to form 2D matrices (time x tmc)
    # Make 2D actual target
    tmc_order_dict = {
        'WB': ['115P04188', '115+04188', '115P04187', '115+04187', '115P04186', '115+04186', '115P04185', '115+04185', '115P04184', '115+04184', '115P04183', '115+04183', '115P04182', '115+04182', '115P04181', '115+04181', '115P04180', '115+04180', '115P04179', '115+04179', '115P04178', '115+04178', '115P04177', '115+04177', '115P05165'],
        'EB': ['115N04188', '115-04187', '115N04187', '115-04186', '115N04186', '115-04185', '115N04185', '115-04184', '115N04184', '115-04183', '115N04183', '115-04182', '115N04182', '115-04181', '115N04181', '115-04180', '115N04180', '115-04179', '115N04179', '115-04178', '115N04178', '115-04177', '115N04177', '115-05165', '115N05165']
    }
    tmc_order = tmc_order_dict[args.dir]
    target_2d = df[TARGET_COL].unstack(level='tmc_code')[tmc_order]

    # Predict over full df
    full_pred_no_evt = pd.Series(model_no_evt.predict(df[no_evt_features]), index=df.index)
    full_pred_plnd = pd.Series(model_planned.predict(df[plnd_evt_features]), index=df.index)
    preds_no_evt_2d = full_pred_no_evt.unstack(level='tmc_code')[tmc_order]
    preds_plnd_2d = full_pred_plnd.unstack(level='tmc_code')[tmc_order]

    # Difference and residuals
    df_diff = preds_plnd_2d - preds_no_evt_2d
    df_residuals = target_2d - preds_plnd_2d
    log(f"Shapes -> target: {getattr(target_2d,'shape',None)}, diff: {getattr(df_diff,'shape',None)}, residuals: {getattr(df_residuals,'shape',None)}")

    # Event matrices
    df_unplnd_evt = df['evt_cat_unplanned'].unstack(level='tmc_code').fillna(0)[tmc_order]
    df_plnd_evt = df['evt_cat_planned'].unstack(level='tmc_code').fillna(0)[tmc_order]

    # 7) Correlation analysis & plots
    df_long = (
        df_diff.stack()
        .rename("diff")
        .to_frame()
        .join(df_residuals.stack().rename("residuals"))
        .join(df_plnd_evt.stack().rename("plnd_evt"))
        .join(df_unplnd_evt.stack().rename("un_plnd_evt"))
        .reset_index()
        .rename(columns={"level_0": "time_bin", "level_1": "tmc"})
    )
    df_long['hour'] = df_long['time_bin'].dt.hour
    df_long['dow'] = df_long['time_bin'].dt.dayofweek
    df_long["event_type"] = np.select(
        [df_long["plnd_evt"] == 1, df_long["un_plnd_evt"] == 1],
        ["planned", "unplanned"],
        default="none"
    )

    # Boxplot: diff by event type
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(x="event_type", y="diff", data=df_long, ax=ax)
    ax.set_ylabel("Estimated Delay (sec/mile)")
    ax.set_title("Distribution of Event-Induced Delay by Event Type")
    savefig(fig, images_dir, 'diff_by_event_type_boxplot')

    # Spatial correlation
    df_long["event_flag"] = (df_long["plnd_evt"] == 1)
    agg_spatial = (
        df_long.groupby("tmc_code")
        .agg(
            mean_diff=("diff", "mean"),
            event_rate=("event_flag", "mean"),
            n_events=("event_flag", "sum")
        )
        .reset_index()
    )
    from scipy.stats import pearsonr, spearmanr
    pearson_corr, _ = pearsonr(agg_spatial["mean_diff"], agg_spatial["event_rate"])
    spearman_corr, _ = spearmanr(agg_spatial["mean_diff"], agg_spatial["event_rate"])
    log(f"Spatial correlation (Pearson mean_diff vs event_rate): {pearson_corr:.3f}")
    log(f"Spatial correlation (Spearman mean_diff vs event_rate): {spearman_corr:.3f}")

    # Temporal correlation and lagged correlation
    agg_temporal = (
        df_long.groupby("time_bin")
        .agg(
            mean_diff=("diff", "mean"),
            event_rate=("event_flag", "mean")
        )
        .reset_index()
    )
    zero_lag_corr, _ = pearsonr(agg_temporal["mean_diff"], agg_temporal["event_rate"])
    log(f"Temporal correlation (zero lag): {zero_lag_corr:.3f}")
    lags = np.arange(-6, 7)
    corr = [
        np.corrcoef(
            np.roll(agg_temporal["event_rate"], l), agg_temporal["mean_diff"]
        )[0, 1]
        for l in lags
    ]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(lags, corr, marker="o")
    ax.axvline(0, color="k", linestyle="--")
    ax.set_title("Correlation between Planned Event Rate and Estimated Delay")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Correlation")
    savefig(fig, images_dir, 'corr_planned_event_rate_vs_delay_lag')

    # Delay due to planned events by hour of day
    def middle_mean(x):
        lower, upper = x.quantile([0.01, 0.99])
        trimmed = x[(x >= lower) & (x <= upper)]
        return trimmed.mean()

    grouped = (
        df_long.groupby(['hour', (df_long['plnd_evt'] > 0)])['diff']
        .apply(middle_mean)
        .fillna(0)
        .unstack(fill_value=0)
        .rename(columns={False: 'no_event', True: 'with_event'})
    )
    grouped_2d = (
        df_long.groupby(['dow', 'hour', (df_long['plnd_evt'] > 0)])['diff']
        .apply(middle_mean)
        .fillna(0)
        .unstack(fill_value=0)
        .rename(columns={False: 'no_event', True: 'with_event'})
    )
    grouped['extra_delay'] = grouped['with_event'] - grouped['no_event']
    grouped_2d['extra_delay'] = grouped_2d['with_event'] - grouped_2d['no_event']

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(grouped.index, grouped['with_event'], label='During planned events', marker='o')
    ax.plot(grouped.index, grouped['no_event'], label='No events', marker='o', linestyle='--')
    ax.bar(grouped.index, grouped['extra_delay'], alpha=0.3, label='Extra delay', color='red')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Estimated Delay (sec/mile)')
    ax.set_title('Average Model-Predicted Delay due to Planned Events by Hour of Day')
    ax.legend()
    savefig(fig, images_dir, 'extra_delay_by_hour')

    # Heatmap (DoW x Hour)
    heatmap_data = grouped_2d['extra_delay'].unstack(level='hour')
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(
        heatmap_data,
        cmap='coolwarm',
        center=0,
        cbar_kws={'label': 'Estimated Delay (sec/mile)'},
        ax=ax
    )
    ax.set_title('Average Model-Predicted Extra Delay due to Planned Events\nby Day of Week and Hour of Day')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Week')
    ax.set_yticks(ticks=np.arange(0.5,7.5,1), labels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    savefig(fig, images_dir, 'extra_delay_heatmap_dow_hour')

    log('Event impact analysis completed.')
    return 0


if __name__ == "__main__":
    sys.exit(main())