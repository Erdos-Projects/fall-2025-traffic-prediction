# Almost equivalent to 'i10_train_time_series.ipynb' but optimized for script execution
# Key differences:
# - Uses 'joblib' with 'multiprocessing' backend for parallelism
# - Configured to use a single CPU core by default (adjustable via 'n_jobs' parameter)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from joblib import Parallel, delayed
import joblib
import gc
import warnings
warnings.filterwarnings("ignore")

# optional: Unix-only per-worker memory cap (bytes)
USE_RLIMIT = True
PER_WORKER_MEM_BYTES = 6 * 1024**3   # e.g., 6 GB per worker

if USE_RLIMIT:
    import resource

# ============== 1. DEFINE EXOG FEATURE SETS ==============
EXOG_CONFIGS = {
    "base": [],
    "evt_total": ["evt_total"],
    "evt": ['evt_cat_minor', 'evt_cat_major','evt_cat_closure','evt_cat_obstruction', 'evt_cat_misc'],
    "cyc": ['hour_sin','hour_cos','dow_sin','dow_cos','hour_of_week_sin','hour_of_week_cos','is_weekend'],
    "full": ['evt_cat_minor', 'evt_cat_major','evt_cat_closure','evt_cat_obstruction', 'evt_cat_misc',
            'hour_sin','hour_cos','dow_sin','dow_cos','hour_of_week_sin','hour_of_week_cos','is_weekend']
}

def safe_set_memlimit(max_bytes):
    """Call inside worker to cap its virtual memory (Unix only)."""
    try:
        # RLIMIT_AS caps the total virtual memory size
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    except Exception as e:
        # not critical, just log
        print("Could not set rlimit:", e)

# train_one_model now takes a pre-sliced task dict and very small return
def train_one_model(task, save_model_path=None, set_rlimit=False, per_worker_mem=None):
    tmc = task["tmc"]
    config_name = task["config"]
    y_train = task["y_train"]
    y_test = task["y_test"]
    X_train = task["X_train"]
    X_test = task["X_test"]

    out = evaluate_sarimax(tmc, config_name,
        y_train, y_test, X_train, X_test,
        order=(1,0,0), seasonal_order=(1,1,1,24),
        save_model_path=save_model_path,
        set_rlimit=set_rlimit,
        per_worker_mem=per_worker_mem
    )
    if out is None:
        return None
    # annotate small metadata and return
    out.update({"TMC": tmc, "Config": config_name})
    return out

def evaluate_sarimax(tmc, config_name, y_train, y_test, X_train, X_test,
                     order=(1,0,0), seasonal_order=(1,1,1,24),
                     save_model_path=None, set_rlimit=False, per_worker_mem=None):
    """Fit SARIMAX and return small metrics. Optionally save fitted model to disk from worker."""
    if set_rlimit and per_worker_mem is not None:
        safe_set_memlimit(per_worker_mem)

    try:
        model = sm.tsa.SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            exog=X_train if X_train is not None and not X_train.empty else None
        )
        res = model.fit(disp=False)
    except Exception as e:
        # return a small error dict so parent can continue
        return {"error": str(e)}

    # Forecast
    forecast = res.get_forecast(steps=len(y_test),
                                exog=X_test if X_test is not None and not X_test.empty else None)
    y_pred = forecast.predicted_mean
    y_pred_in = res.fittedvalues

    # scaled metrics
    eps = 1e-6
    def scaled_metrics(y_true, y_pred_):
        mape = np.mean(np.abs((y_true - y_pred_) / (y_true + eps))) * 100
        smape = 100 * np.mean(2 * np.abs(y_true - y_pred_) / (np.abs(y_true) + np.abs(y_pred_) + eps))
        nrmse = np.sqrt(np.mean((y_true - y_pred_)**2)) / (np.mean(y_true) + eps)
        corr = np.corrcoef(y_true, y_pred_)[0,1]
        return mape, smape, nrmse, corr

    mape_train, smape_train, nrmse_train, corr_train = scaled_metrics(y_train, y_pred_in)
    mape_test, smape_test, nrmse_test, corr_test = scaled_metrics(y_test, y_pred)

    metrics = {
        "AIC": float(np.nan if np.isinf(np.nan) else getattr(res, "aic", np.nan)),  # keep placeholder if needed
        "BIC": float(getattr(res, "bic", np.nan)),
        "MAPE_train": mape_train,
        "MAPE_test": mape_test,
        "NRMSE_train": nrmse_train,
        "NRMSE_test": nrmse_test,
        "corr_train": corr_train,
        "corr_test": corr_test,
        "model_summary": res.summary(),       # small string or None
        "converged": bool(res.mle_retvals.get("converged", False))
    }
    # Optionally save the model from inside worker (avoid serializing back)
    model_path = None
    if save_model_path is not None:
        try:
            # create dir if not exists
            os.makedirs(save_model_path, exist_ok=True)
            # unique filename
            # fname = f"model_{int(datetime.utcnow().timestamp() * 1000)}.pkl"
            fname = f"sarimax_{tmc}_{config_name}.pkl"
            model_path = os.path.join(save_model_path, fname)
            # strip heavy intermediate arrays
            res.save(model_path, remove_data=True)
            print("Model saved to:", model_path)
        except Exception as e:
            # don't fail whole job if saving fails
            model_path = None

    # IMPORTANT: delete large objects before returning
    try:
        del res
        del model
        gc.collect()
    except Exception:
        pass

    return metrics

# ============== 2. MODEL TRAINING + EVALUATION FUNCTION ==============
def scaled_metrics(y_true, y_pred):
    eps = 1e-6
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps))
    nrmse = np.sqrt(np.mean((y_true - y_pred)**2)) / (np.mean(y_true) + eps)
    corr = np.corrcoef(y_true, y_pred)[0,1]
    return mape, smape, nrmse, corr


# ============== 3. FUNCTION TO TRAIN ONE MODEL CONFIGURATION ==============
def run_tasks_in_batches(tasks, n_jobs=1, batch_size=20, save_model_path=None,
                        set_rlimit=False, per_worker_mem=PER_WORKER_MEM_BYTES):
    results = []

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        print(f"Running batch {i//batch_size + 1} / {int(np.ceil(len(tasks)/batch_size))} with n_jobs={n_jobs}")
        batch_res = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=10)(
            delayed(train_one_model)(t, save_model_path, set_rlimit, per_worker_mem) for t in batch
        )
        # filter
        batch_res = [r for r in batch_res if r is not None]
        results.extend(batch_res)

    return pd.DataFrame(results)

# ============== 4. PARALLEL MODEL COMPARISON ==============
def run_model_comparison_parallel(X_full, target_col="travel_time_seconds", save_model_path=None, tmc_list=None, n_jobs=None):
    if tmc_list is None:
        tmc_list = X_full.index.get_level_values("tmc_code").unique()

    if n_jobs is None:
        n_jobs = max(1, 64)

    print(f"Running SARIMAX comparison on {len(tmc_list)} TMCs using {n_jobs} cores ...")

    tasks = []
    for tmc in tmc_list:
        grp = X_full.xs(tmc, level="tmc_code").sort_index(level="time_bin")

        n_test = int(len(grp) * 0.2)
        y_train = grp[target_col].iloc[:-n_test].values
        y_test  = grp[target_col].iloc[-n_test:].values

        for config_name, exog_cols in EXOG_CONFIGS.items():
            if exog_cols:
                X_train = grp[exog_cols].iloc[:-n_test].copy()
                X_test  = grp[exog_cols].iloc[-n_test:].copy()
            else:
                X_train, X_test = None, None

            tasks.append({
                "tmc": tmc,
                "config": config_name,
                "y_train": y_train,
                "y_test": y_test,
                "X_train": X_train,
                "X_test": X_test
            })

    
    df_results = run_tasks_in_batches(tasks, n_jobs=n_jobs, batch_size=20, save_model_path=save_model_path,
                        set_rlimit=True, per_worker_mem=PER_WORKER_MEM_BYTES)
    return df_results
    
def main():

    # load data
    data_path = Path('database/i10-broadway')
    X_full = pd.read_parquet(data_path / 'X_full_1h.parquet')
    print('X_full type:', type(X_full), 'shape:', getattr(X_full, 'shape', None))

    # ============== 5. RUN PARALLELIZED COMPARISON ==============
    tmc_list = X_full.index.get_level_values("tmc_code").unique()
    df_results = run_model_comparison_parallel(
        X_full,
        target_col="travel_time_seconds",
        save_model_path=None, #'models/sarimax/',
        tmc_list=tmc_list,
        n_jobs=8  # adjust to CPU
    )

    # ============== 6. SAVE RESULTS ==============
    joblib.dump(df_results, 'models/sarimax/sarimax_results.pkl')

    return 


if __name__ == "__main__":
    main()
 