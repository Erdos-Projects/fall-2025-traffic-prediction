"""
prepare_i10_training_data.py
Author: Yanbing Wang
Date: 2025-10-23

A clean, modular CLI for preparing I-10 Broadway Curve training data by combining:
- AZ511 events from a SQLite database
- INRIX speeds from CSV (with TMC metadata)

It filters to a bounding box around the Broadway Curve, assigns each event to the nearest
TMC (direction-aware), aggregates INRIX to a time grid, joins event overlaps, engineers
time and lag features, and saves a parquet dataset.

Outputs (default names under out-dir):
- events.parquet, inrix.parquet, tmc.parquet (optional, for inspection)
- X_full_<interval>.parquet with MultiIndex (tmc_code, time_bin)

Key columns used downstream (e.g., train_model_* scripts):
- Target: tt_per_mile
- Time features: hour_sin, hour_cos, dow_sin, dow_cos, hour_of_week_sin, hour_of_week_cos, is_weekend
- Event features: evt_cat_unplanned, evt_cat_planned
- Lag features: lag1_tt_per_mile, lag2_tt_per_mile, lag3_tt_per_mile
- TMC static: miles, reference_speed, curve, onramp, offramp

Notes:
- This script is intentionally tailored to the I-10 Broadway Curve. Bounds and
    road name are parameterized for convenience.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import re
import sqlite3
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Utilities
# -------------------------
def _to_datetime_utc(series: pd.Series) -> pd.Series:
    """Robustly convert a Series to UTC timestamps.
    - Numeric values are interpreted as seconds or milliseconds since epoch (heuristic by magnitude).
    - Datetime-like values are converted or localized to UTC.
    """
    if series is None:
        return pd.Series(pd.NaT, index=pd.RangeIndex(0))
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        non_na = s.dropna()
        if non_na.empty:
            return pd.to_datetime(s, errors="coerce", utc=True)
        median_val = float(non_na.median())
        unit = "ms" if median_val > 1e12 else "s"
        return pd.to_datetime(s, unit=unit, errors="coerce", utc=True)
    # datetime-like or strings
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt


def _ensure_utc(ts: pd.Series) -> pd.Series:
    """Ensure a datetime Series is tz-aware in UTC.
    If tz-naive: localize to UTC. If tz-aware: convert to UTC.
    """
    dt = pd.to_datetime(ts, errors="coerce", utc=False)
    tzinfo = getattr(dt.dt, "tz", None)
    if tzinfo is None:
        return dt.dt.tz_localize("UTC")
    return dt.dt.tz_convert("UTC")


# -------------------------
# AZ511 events
# -------------------------
def load_az511_events(db_path: Path, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print(f"Reading AZ511 events from {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM events", conn)
    # Normalize core time columns to UTC
    for col in ["Reported", "LastUpdated", "StartDate", "PlannedEndDate"]:
        if col in df.columns:
            df[col] = _to_datetime_utc(df[col])
    return df


def filter_broadway_events(
    events: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    road_pattern: str = r"I-10|I10",
    verbose: bool = True,
) -> pd.DataFrame:
    m_geo = (
        events["Latitude"].between(lat_min, lat_max)
        & events["Longitude"].between(lon_min, lon_max)
    )
    m_road = events["RoadwayName"].astype(str).str.contains(road_pattern, case=False, na=False)
    df = events.loc[m_geo & m_road].copy()

    # Infer direction if 'Unknown'
    if "DirectionOfTravel" in df.columns and "RoadwayName" in df.columns:
        is_unknown = df["DirectionOfTravel"].astype(str).str.lower().eq("unknown")
        inferred = (
            df.loc[is_unknown, "RoadwayName"]
            .astype(str)
            .str.extract(r"(west|east|south|north)", flags=re.I, expand=False)
            .str.lower()
            .fillna("unknown")
        )
        df.loc[is_unknown, "DirectionOfTravel"] = inferred

    # Drop heavy rarely-used columns if present
    drop_cols = [
        "EncodedPolyline",
        "Width",
        "Height",
        "Length",
        "Weight",
        "Speed",
        "DetourPolyline",
        "DetourInstructions",
    ]
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    if verbose:
        print(f"Filtered events in Broadway area: {df.shape}")
        if "EventType" in df.columns:
            print(df["EventType"].value_counts().head())
    return df


# -------------------------
# TMC metadata
# -------------------------
def load_tmc_metadata(
    tmc_csv: Path,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    road_name: str = "I-10",
    verbose: bool = True,
) -> pd.DataFrame:
    usecols = [
        "tmc",
        "road",
        "direction",
        "intersection",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
        "miles",
        "road_order",
        "active_start_date",
    ]
    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        tmc_csv,
        usecols=usecols,
        chunksize=50_000,
        dtype={
            "tmc": "string",
            "road": "string",
            "direction": "string",
            "intersection": "string",
            "miles": "float32",
            "road_order": "float32",
        },
    ):
        m = (
            (
                chunk.start_latitude.between(lat_min, lat_max)
                & chunk.start_longitude.between(lon_min, lon_max)
            )
            | (
                chunk.end_latitude.between(lat_min, lat_max)
                & chunk.end_longitude.between(lon_min, lon_max)
            )
        ) & (chunk["road"] == road_name)
        if m.any():
            chunks.append(chunk.loc[m])
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)

    # Deduplicate TMC by keeping latest active_start_date
    if "active_start_date" in df.columns:
        df["active_start_date"] = pd.to_datetime(df["active_start_date"], errors="coerce")
        df = (
            df.sort_values(["tmc", "active_start_date"], na_position="first")
            .groupby("tmc", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
    if verbose:
        print("TMC rows in bounds:", len(df))
    return df


# -------------------------
# INRIX
# -------------------------
def load_inrix_filtered(
    inrix_csv: Path,
    tmc_set: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    verbose: bool = True,
) -> pd.DataFrame:
    usecols = [
        "tmc_code",
        "measurement_tstamp",
        "speed",
        "reference_speed",
        "travel_time_seconds",
        "confidence_score",
        "cvalue",
        "Inrix 2013",
        "Inrix 2019",
    ]
    tmc_set = set(tmc_set)
    parts: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        inrix_csv, usecols=usecols, parse_dates=["measurement_tstamp"], chunksize=20_000
    ):
        # Ensure measurement_tstamp is tz-aware (UTC) to match start/end
        chunk["measurement_tstamp"] = _ensure_utc(chunk["measurement_tstamp"])
        m = (
            chunk["tmc_code"].isin(tmc_set)
            & (chunk["measurement_tstamp"] >= start)
            & (chunk["measurement_tstamp"] <= end)
        )
        if m.any():
            parts.append(chunk.loc[m])
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
    if verbose:
        print("Filtered INRIX rows:", len(df))
    return df


def aggregate_inrix(
    df_inrix: pd.DataFrame, interval: str = "1h", time_col: str = "measurement_tstamp"
) -> Tuple[pd.DataFrame, List[str]]:
    """Aggregate INRIX to [tmc_code, time_bin] with means for numeric fields.
    Returns (X_df, value_cols_used)
    """
    drop_cols = ["confidence_score", "cvalue", "Inrix 2013", "Inrix 2019"]
    df = df_inrix.drop(columns=[c for c in drop_cols if c in df_inrix.columns], errors="ignore").copy()

    group_col = "tmc_code"
    if time_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"INRIX must contain '{time_col}' and '{group_col}'")

    # Ensure UTC and floor to bins
    dt = _ensure_utc(df[time_col])
    df["time_bin"] = dt.dt.floor(interval)
    df = df.dropna(subset=["time_bin", group_col])

    value_cols = [c for c in ["speed", "travel_time_seconds", "reference_speed"] if c in df.columns]
    agg_map = {c: "mean" for c in value_cols}
    other_cols = [c for c in df.columns if c not in set(value_cols + [time_col, group_col, "time_bin"])]
    for c in other_cols:
        agg_map[c] = "first"

    X = (
        df.groupby([group_col, "time_bin"], as_index=False)
        .agg(agg_map)
        .sort_values([group_col, "time_bin"])\
        .set_index([group_col, "time_bin"])\
        .sort_index()
    )
    return X, value_cols


# -------------------------
# Event-to-TMC assignment
# -------------------------
def _points_segments_nearest(events_xy: np.ndarray, seg_a_xy: np.ndarray, seg_b_xy: np.ndarray):
    N = events_xy.shape[0]
    M = seg_a_xy.shape[0]
    v = seg_b_xy - seg_a_xy
    v_len2 = (v ** 2).sum(axis=1)
    v_len2[v_len2 == 0] = 1e-12

    chunk = 500
    best_dist2 = np.full(N, np.inf)
    best_idx = np.zeros(N, dtype=int)
    best_t = np.zeros(N)

    P = events_xy
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        A = seg_a_xy[s:e]
        V = v[s:e]
        L2 = v_len2[s:e]

        PA = P[:, None, :] - A[None, :, :]
        t = (PA * V[None, :, :]).sum(axis=2) / L2[None, :]
        t_clamped = np.clip(t, 0, 1)

        proj = A[None, :, :] + t_clamped[..., None] * V[None, :, :]
        diff = P[:, None, :] - proj
        dist2 = (diff ** 2).sum(axis=2)

        local_min_idx = dist2.argmin(axis=1)
        local_min_val = dist2[np.arange(N), local_min_idx]
        improved = local_min_val < best_dist2
        if improved.any():
            best_dist2[improved] = local_min_val[improved]
            best_idx[improved] = s + local_min_idx[improved]
            best_t[improved] = t_clamped[np.arange(N), local_min_idx][improved]

    proj_xy = seg_a_xy[best_idx] + best_t[:, None] * v[best_idx]
    return best_idx, np.sqrt(best_dist2), proj_xy, best_t


def assign_nearest_tmc_direction_aware(df_events_bw: pd.DataFrame, df_tmc: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if df_events_bw.empty or df_tmc.empty:
        return df_events_bw

    lat0 = float(df_events_bw["Latitude"].mean())
    m_per_deg_lat = 110_540.0
    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))

    df = df_events_bw.copy()
    df["DirectionOfTravel"] = df["DirectionOfTravel"].astype(str).str.lower()

    dir_map = {
        "east": "EASTBOUND",
        "west": "WESTBOUND",
        "north": "NORTHBOUND",
        "south": "SOUTHBOUND",
    }

    df["near_tmc"] = pd.NA
    base_mask = df[["Latitude", "Longitude"]].notna().all(axis=1)

    for ev_dir, tmc_dir in dir_map.items():
        ev_mask_dir = base_mask & df["DirectionOfTravel"].eq(ev_dir)
        if not ev_mask_dir.any():
            continue
        seg_mask = df_tmc["direction"].astype(str).str.contains(tmc_dir, case=False, na=False)
        seg_subset = df_tmc.loc[seg_mask]
        if seg_subset.empty:
            continue

        ev_lat = df.loc[ev_mask_dir, "Latitude"].to_numpy()
        ev_lon = df.loc[ev_mask_dir, "Longitude"].to_numpy()
        events_xy = np.column_stack([ev_lon * m_per_deg_lon, ev_lat * m_per_deg_lat])

        seg_a_xy = np.column_stack([
            seg_subset["start_longitude"].to_numpy() * m_per_deg_lon,
            seg_subset["start_latitude"].to_numpy() * m_per_deg_lat,
        ])
        seg_b_xy = np.column_stack([
            seg_subset["end_longitude"].to_numpy() * m_per_deg_lon,
            seg_subset["end_latitude"].to_numpy() * m_per_deg_lat,
        ])

        idx_local, _, _, _ = _points_segments_nearest(events_xy, seg_a_xy, seg_b_xy)
        df.loc[ev_mask_dir, "near_tmc"] = seg_subset.iloc[idx_local]["tmc"].astype("string").to_numpy()

    # Fallback for unmatched/unknown
    unmatched = base_mask & df["near_tmc"].isna()
    if unmatched.any():
        seg_a_xy_all = np.column_stack([
            df_tmc["start_longitude"].to_numpy() * m_per_deg_lon,
            df_tmc["start_latitude"].to_numpy() * m_per_deg_lat,
        ])
        seg_b_xy_all = np.column_stack([
            df_tmc["end_longitude"].to_numpy() * m_per_deg_lon,
            df_tmc["end_latitude"].to_numpy() * m_per_deg_lat,
        ])
        ev_lat = df.loc[unmatched, "Latitude"].to_numpy()
        ev_lon = df.loc[unmatched, "Longitude"].to_numpy()
        events_xy = np.column_stack([ev_lon * m_per_deg_lon, ev_lat * m_per_deg_lat])
        idx_all, _, _, _ = _points_segments_nearest(events_xy, seg_a_xy_all, seg_b_xy_all)
        df.loc[unmatched, "near_tmc"] = df_tmc.iloc[idx_all]["tmc"].astype("string").to_numpy()

    return df


# -------------------------
# Manual TMC tags
# -------------------------
def add_manual_tmc_tags(df_tmc: pd.DataFrame) -> pd.DataFrame:
    road_tags: Dict[str, Dict[str, int]] = {
        # Westbound
        '115P04188': {'onramp': 1, 'offramp': 0, 'curve': 1},
        '115+04188': {'onramp': 0, 'offramp': 1, 'curve': 1},
        '115P04187': {'onramp': 0, 'offramp': 1, 'curve': 1},
        '115+04187': {'onramp': 0, 'offramp': 1, 'curve': 1},
        '115P04186': {'onramp': 1, 'offramp': 0, 'curve': 1},
        '115+04186': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115P04185': {'onramp': 1, 'offramp': 0, 'curve': 0},
        '115+04185': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115P04184': {'onramp': 2, 'offramp': 0, 'curve': 1},
        '115+04184': {'onramp': 0, 'offramp': 1, 'curve': 1},
        '115P04183': {'onramp': 0, 'offramp': 0, 'curve': 1},
        '115+04183': {'onramp': 0, 'offramp': 1, 'curve': 1},
        '115P04182': {'onramp': 2, 'offramp': 0, 'curve': 0},
        '115+04182': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115P04181': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115+04181': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115P04180': {'onramp': 1, 'offramp': 0, 'curve': 0},
        '115+04180': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115P04179': {'onramp': 1, 'offramp': 0, 'curve': 0},
        '115+04179': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115P04178': {'onramp': 1, 'offramp': 0, 'curve': 0},
        '115+04178': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115P04177': {'onramp': 2, 'offramp': 0, 'curve': 0},
        '115+04177': {'onramp': 0, 'offramp': 0, 'curve': 0},
        '115P05165': {'onramp': 0, 'offramp': 0, 'curve': 0},
        # Eastbound
        '115N04188': {'onramp': 1, 'offramp': 0, 'curve': 1},
        '115-04187': {'onramp': 0, 'offramp': 0, 'curve': 0},
        '115N04187': {'onramp': 1, 'offramp': 0, 'curve': 1},
        '115-04186': {'onramp': 1, 'offramp': 2, 'curve': 1},
        '115N04186': {'onramp': 1, 'offramp': 0, 'curve': 1},
        '115-04185': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115N04185': {'onramp': 1, 'offramp': 0, 'curve': 0},
        '115-04184': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115N04184': {'onramp': 2, 'offramp': 0, 'curve': 1},
        '115-04183': {'onramp': 0, 'offramp': 0, 'curve': 1},
        '115N04183': {'onramp': 0, 'offramp': 0, 'curve': 1},
        '115-04182': {'onramp': 0, 'offramp': 2, 'curve': 0},
        '115N04182': {'onramp': 1, 'offramp': 1, 'curve': 0},
        '115-04181': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115N04181': {'onramp': 2, 'offramp': 0, 'curve': 0},
        '115-04180': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115N04180': {'onramp': 1, 'offramp': 0, 'curve': 0},
        '115-04179': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115N04179': {'onramp': 1, 'offramp': 0, 'curve': 0},
        '115-04178': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115N04178': {'onramp': 1, 'offramp': 0, 'curve': 0},
        '115-04177': {'onramp': 0, 'offramp': 1, 'curve': 0},
        '115N04177': {'onramp': 1, 'offramp': 0, 'curve': 0},
        '115-05165': {'onramp': 0, 'offramp': 0, 'curve': 0},
        '115N05165': {'onramp': 1, 'offramp': 0, 'curve': 0},
    }
    df = df_tmc.copy()
    df["tmc_tags"] = df["tmc"].map(road_tags)
    return df


# -------------------------
# Event features over time grid
# -------------------------
EVENT_MAPPING = {
    # 1. Major Lane-Blocking
    "crashRlane": "cat_unplanned",
    "crashLlane": "cat_unplanned",
    "crashHOV": "cat_unplanned",
    "C34Rlane": "cat_unplanned",
    "C34leftLanes": "cat_unplanned",
    "C34HOVlane": "cat_unplanned",
    "Freewayclosed": "cat_unplanned",
    "RoadClosedDueToCrash": "cat_unplanned",
    "AccidentIncident": "cat_unplanned",
    "accident": "cat_unplanned",
    "CrashMedian": "cat_unplanned",
    "crashBIntersection": "cat_unplanned",
    # 2. Shoulder/Ramp Incidents
    "C34Rshoulder": "cat_unplanned",
    "C34Lshoulder": "cat_unplanned",
    "Crash on right shoulder": "cat_unplanned",
    "CrashLshoulder": "cat_unplanned",
    "DebrisRshoulder": "cat_unplanned",
    "debrisLshoulder": "cat_unplanned",
    "vehicleOnFire": "cat_unplanned",
    "pedestrianOnRoadway": "cat_unplanned",
    "animalOnRoadway": "cat_unplanned",
    "C34exit": "cat_unplanned",
    "TC34EXB": "cat_unplanned",
    "C34onramp": "cat_unplanned",
    "crashBOnramp": "cat_unplanned",
    "crashExit": "cat_unplanned",
    "crashBExit": "cat_unplanned",
    "crashOnramp": "cat_unplanned",
    "TC34ONB": "cat_unplanned",
    # 3. Planned
    "leftlanes": "cat_planned",
    "rightlanes": "cat_planned",
    "LeftLane": "cat_planned",
    "rightlane": "cat_planned",
    "shoulderclosed": "cat_planned",
    "exitrestricted": "cat_planned",
    "exitclosed": "cat_planned",
    "hovrampclosed": "cat_planned",
    "ramp2westclosed": "cat_planned",
    "ramp2eastclosed": "cat_planned",
    "patching": "cat_planned",
    # 4. Debris/obstruction
    "debrisLlane": "cat_unplanned",
    "debrisRlane": "cat_unplanned",
    "debrisinroad": "cat_unplanned",
    "debrisClane": "cat_unplanned",
    "Potholes": "cat_unplanned",
    "DeadAnimalRoad": "cat_unplanned",
    "Graffiti": "cat_unplanned",
    "guardrail": "cat_unplanned",
    "fencedamage": "cat_unplanned",
    # 5. Misc
    "signaltiming": "cat_unplanned",
    "SignalIssue": "cat_unplanned",
    "SignalRedbulb": "cat_unplanned",
    "signdamaged": "cat_unplanned",
    "ITS Equipment Damage": "cat_unplanned",
    "TDMG": "cat_unplanned",
    "T1018SR": "cat_unplanned",
    "TDMGLEAK": "cat_unplanned",
    "TDEBRISRM": "cat_unplanned",
    "TDEBRISLM": "cat_unplanned",
    "TDEBRISCM": "cat_unplanned",
    "TDEBRISONB": "cat_unplanned",
    "TC34I": "cat_unplanned",
}


def build_event_features(
    X_index: pd.MultiIndex,
    events_bw: pd.DataFrame,
    interval: str,
) -> Tuple[pd.DataFrame, List[str]]:
    group_col = "tmc_code"
    tmcs = X_index.get_level_values(group_col).unique().sort_values()
    times = X_index.get_level_values("time_bin").unique().sort_values()

    # Normalize event windows
    etype = events_bw["EventType"].astype(str).str.strip().str.lower()
    s_start = _ensure_utc(events_bw.get("StartDate")) if "StartDate" in events_bw else pd.Series(pd.NaT, index=events_bw.index)
    s_end = _ensure_utc(events_bw.get("PlannedEndDate")) if "PlannedEndDate" in events_bw else pd.Series(pd.NaT, index=events_bw.index)
    s_rep = _ensure_utc(events_bw.get("Reported")) if "Reported" in events_bw else pd.Series(pd.NaT, index=events_bw.index)
    s_upd = _ensure_utc(events_bw.get("LastUpdated")) if "LastUpdated" in events_bw else pd.Series(pd.NaT, index=events_bw.index)

    is_plan = etype.isin(["closures", "roadwork"])
    is_inc = etype.eq("accidentsandincidents")

    event_start = pd.Series(pd.NaT, index=events_bw.index, dtype="datetime64[ns, UTC]")
    event_end = pd.Series(pd.NaT, index=events_bw.index, dtype="datetime64[ns, UTC]")

    event_start[is_plan] = s_start[is_plan]
    event_end[is_plan] = s_end[is_plan]
    event_start[is_inc] = s_rep[is_inc]
    event_end[is_inc] = s_upd[is_inc]
    event_end = event_end.fillna(event_start)

    evt_cat_levels = ["cat_unplanned", "cat_planned"]
    events_proc = pd.DataFrame(
        {
            "near_tmc_str": events_bw["near_tmc"].astype(str),
            "event_start": event_start,
            "event_end": event_end,
            "cat": events_bw.get("cat_event_type", "cat_unplanned").astype(str),
        }
    ).dropna(subset=["near_tmc_str", "event_start", "event_end"])
    events_proc["cat"] = events_proc["cat"].where(events_proc["cat"].isin(evt_cat_levels), "cat_unplanned")

    evt_cols = [f"evt_{c}" for c in evt_cat_levels]
    evt_parts: List[pd.DataFrame] = []
    for tmc in tmcs:
        b_start = pd.DatetimeIndex(times)
        b_end = b_start + pd.to_timedelta(interval)
        edf = events_proc[events_proc["near_tmc_str"] == str(tmc)]
        if edf.empty:
            feat = pd.DataFrame(0, index=b_start, columns=evt_cols)
        else:
            e_start = edf["event_start"].to_numpy()
            e_end = edf["event_end"].to_numpy()
            cat_vals = edf["cat"].to_numpy()
            bs = b_start.to_numpy()[:, None]
            be = b_end.to_numpy()[:, None]
            overlap = (bs < e_end[None, :]) & (be > e_start[None, :])
            data = {}
            for c in evt_cat_levels:
                mask_c = cat_vals[None, :] == c
                data[f"evt_{c}"] = (overlap & mask_c).sum(axis=1).astype(int)
            feat = pd.DataFrame(data, index=b_start)
        feat.index.name = "time_bin"
        feat[group_col] = tmc
        feat = feat.reset_index().set_index([group_col, "time_bin"]).sort_index()
        evt_parts.append(feat)

    evt_df = pd.concat(evt_parts).sort_index() if evt_parts else pd.DataFrame(0, index=X_index, columns=evt_cols)
    return evt_df, evt_cols


def add_time_and_lag_features(X_full: pd.DataFrame) -> pd.DataFrame:
    t = X_full.index.get_level_values("time_bin")
    X_full = X_full.assign(
        hour_sin=np.sin(2 * np.pi * t.hour / 24),
        hour_cos=np.cos(2 * np.pi * t.hour / 24),
        dow_sin=np.sin(2 * np.pi * t.dayofweek / 7),
        dow_cos=np.cos(2 * np.pi * t.dayofweek / 7),
        hour_of_week=t.dayofweek * 24 + t.hour,
        hour_of_week_sin=np.sin(2 * np.pi * (t.dayofweek * 24 + t.hour) / (7 * 24)),
        hour_of_week_cos=np.cos(2 * np.pi * (t.dayofweek * 24 + t.hour) / (7 * 24)),
        is_weekend=(t.dayofweek >= 5).astype(int),
    )
    # Lags on travel_time_seconds
    for lag in (1, 2, 3):
        X_full[f"lag{lag}"] = (
            X_full.groupby(level="tmc_code")["travel_time_seconds"].shift(lag)
        )
    # Drop rows without full lag history
    X_full = X_full.dropna(subset=["lag1", "lag2", "lag3"])  # cautious: trims start of each TMC

    # Targets
    X_full["tt_per_mile"] = X_full["travel_time_seconds"] / X_full["miles"]
    X_full["lag1_tt_per_mile"] = X_full["lag1"] / X_full["miles"]
    X_full["lag2_tt_per_mile"] = X_full["lag2"] / X_full["miles"]
    X_full["lag3_tt_per_mile"] = X_full["lag3"] / X_full["miles"]
    # Derived
    if {"speed", "reference_speed"}.issubset(set(X_full.columns)):
        X_full["speed_ratio"] = X_full["speed"] / X_full["reference_speed"]
    return X_full


# -------------------------
# Main pipeline
# -------------------------
def run_pipeline(
    db_path: Path,
    tmc_csv: Path,
    inrix_csv: Path,
    out_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str = "1h",
    lat_min: float = 33.296690,
    lat_max: float = 33.428422,
    lon_min: float = -112.039731,
    lon_max: float = -111.962382,
    road_name: str = "I-10",
    write_intermediate: bool = True,
    verbose: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & filter events
    events = load_az511_events(db_path, verbose=verbose)
    events_bw = filter_broadway_events(events, lat_min, lat_max, lon_min, lon_max, road_pattern=road_name, verbose=verbose)

    # 2) TMC metadata and nearest assignment
    df_tmc = load_tmc_metadata(tmc_csv, lat_min, lat_max, lon_min, lon_max, road_name=road_name, verbose=verbose)
    events_bw = assign_nearest_tmc_direction_aware(events_bw, df_tmc, verbose=verbose)
    df_tmc = add_manual_tmc_tags(df_tmc)

    # 3) INRIX
    df_inrix = load_inrix_filtered(inrix_csv, set(df_tmc["tmc"].astype(str)), start, end, verbose=verbose)
    X_inrix, value_cols = aggregate_inrix(df_inrix, interval=interval)

    # 4) Event categories
    if "EventSubType" in events_bw:
        events_bw["cat_event_type"] = events_bw["EventSubType"].map(EVENT_MAPPING).fillna("cat_unplanned")
    else:
        events_bw["cat_event_type"] = "cat_unplanned"

    # 5) Event features aligned to grid
    evt_df, evt_cols = build_event_features(X_inrix.index, events_bw, interval)
    X = X_inrix.join(evt_df, how="right")
    for c in evt_cols:
        if c not in X:
            X[c] = 0
    X[evt_cols] = X[evt_cols].fillna(0)

    # 6) Fill miles and manual tags
    miles_map = df_tmc.drop_duplicates("tmc").set_index("tmc")["miles"].astype(float)
    X["miles"] = X.index.get_level_values("tmc_code").map(miles_map)

    # Manual tags expanded as columns
    tmc_base = df_tmc.drop_duplicates("tmc").set_index("tmc")
    tag_cols: List[str] = []
    if "tmc_tags" in tmc_base.columns:
        tags_norm = (
            tmc_base["tmc_tags"].apply(lambda v: v if isinstance(v, dict) else {}).apply(pd.Series).fillna(0)
        )
        tags_norm = tags_norm.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        tag_cols = tags_norm.columns.tolist()
        mapping_dict = {col: tags_norm[col].to_dict() for col in tag_cols}
        tmc_idx = X.index.get_level_values("tmc_code")
        for col in tag_cols:
            X[col] = tmc_idx.map(mapping_dict[col]).fillna(0).astype(int)

    # 7) Interpolate per TMC and add time/lag features
    X = (
        X.groupby(level="tmc_code")
        .apply(lambda g: g.sort_index(level="time_bin").interpolate(method="linear", limit_direction="both"))
        .droplevel(0)
    )

    # Derived simple event combo
    evt_count_cols = [c for c in ["evt_cat_unplanned", "evt_cat_planned"] if c in X.columns]
    if evt_count_cols:
        X["evt_total"] = X[evt_count_cols].sum(axis=1)

    X = add_time_and_lag_features(X)

    # 8) Save outputs
    if write_intermediate:
        events_bw.to_parquet(out_dir / "events.parquet", index=False)
        df_inrix.to_parquet(out_dir / "inrix.parquet", index=False)
        df_tmc.to_parquet(out_dir / "tmc.parquet", index=False)
        if verbose:
            print("Saved intermediate parquet files to", out_dir)

    out_path = out_dir / f"X_full_{interval}.parquet"
    X.to_parquet(out_path)
    if verbose:
        n_pairs = X.index.nunique()
        print(f"✅ Wrote {out_path} with shape {X.shape} and {n_pairs} unique (tmc,time) pairs")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare I-10 Broadway Curve training dataset (events + INRIX)")
    p.add_argument("--db-path", type=Path, default=Path("database/az511.db"), help="Path to AZ511 SQLite database")
    p.add_argument(
        "--tmc-csv",
        type=Path,
        default=Path("database/inrix-traffic-speed/I10-and-I17-1year/TMC_Identification.csv"),
        help="Path to INRIX TMC_Identification.csv",
    )
    p.add_argument(
        "--inrix-csv",
        type=Path,
        default=Path("database/inrix-traffic-speed/I10-and-I17-1year/I10-and-I17-1year.csv"),
        help="Path to INRIX CSV file",
    )
    p.add_argument("--out-dir", type=Path, default=Path("database/i10-broadway"), help="Output directory for parquet files")
    p.add_argument("--start", type=str, default="2025-06-16T00:00:00Z", help="Start datetime (ISO, e.g., 2025-06-16T00:00:00Z)")
    p.add_argument("--end", type=str, default="2025-09-23T00:00:00Z", help="End datetime (ISO)")
    p.add_argument("--interval", type=str, default="1h", help="Aggregation interval for INRIX (e.g., 5min, 15min, 1h)")
    p.add_argument("--lat-min", type=float, default=33.296690, help="Latitude min bound")
    p.add_argument("--lat-max", type=float, default=33.428422, help="Latitude max bound")
    p.add_argument("--lon-min", type=float, default=-112.039731, help="Longitude min bound")
    p.add_argument("--lon-max", type=float, default=-111.962382, help="Longitude max bound")
    p.add_argument("--road-name", type=str, default="I-10", help="Road name filter pattern (regex, e.g., I-10|I10)")
    p.add_argument("--no-intermediate", action="store_true", help="Do not write intermediate parquet files")
    p.add_argument("--quiet", action="store_true", help="Reduce logging")
    return p.parse_args()


def main():
    # Avoid MKL duplicate warnings in some envs
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    args = parse_args()
    start = pd.to_datetime(args.start, utc=True)
    end = pd.to_datetime(args.end, utc=True)
    verbose = not args.quiet
    write_intermediate = not args.no_intermediate

    if verbose:
        print("Config:")
        print(" - db_path:", args.db_path)
        print(" - tmc_csv:", args.tmc_csv)
        print(" - inrix_csv:", args.inrix_csv)
        print(" - out_dir:", args.out_dir)
        print(" - start/end:", start, end)
        print(" - interval:", args.interval)

    run_pipeline(
        db_path=args.db_path,
        tmc_csv=args.tmc_csv,
        inrix_csv=args.inrix_csv,
        out_dir=args.out_dir,
        start=start,
        end=end,
        interval=args.interval,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        road_name=args.road_name,
        write_intermediate=write_intermediate,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
