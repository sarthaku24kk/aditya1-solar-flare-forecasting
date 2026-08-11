"""Turn new raw readings (uploaded CSV) into model-ready features and a forecast.

The input CSV must have a time column and at least one channel column.
Column mapping is fuzzy: solexs* -> SoLEXS channel, hel1os_<det>_<elow>-<ehigh>keV
channels are matched by detector name. Missing channels are filled with the
training-set median (neutral value) so predictions still work.
"""
from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH
from src.features import build_features, sample_rows

TIME_COL_HINTS = ["time", "timestamp", "date", "utc", "isodatetime", "tstart"]


def _find_time_col(columns: list[str]) -> str | None:
    for c in columns:
        if c.strip().lower() in TIME_COL_HINTS:
            return c
    for c in columns:
        if "time" in c.lower() or "date" in c.lower():
            return c
    return None


def load_uploaded(file_bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV into a DataFrame with a datetime index."""
    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))
    if df.empty:
        raise ValueError("The uploaded file is empty.")
    time_col = _find_time_col(list(df.columns))
    if time_col is None:
        raise ValueError(
            "No time column found. Expected one of: " + ", ".join(TIME_COL_HINTS)
        )
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    if ts.isna().all():
        raise ValueError(f"Could not parse the time column '{time_col}'.")
    out = df.drop(columns=[time_col]).copy()
    out.index = ts.dt.tz_localize(None)
    return out.dropna(subset=[out.columns[0]]) if len(out.columns) else out


def _map_columns(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Rename uploaded columns to the canonical feature-channel names."""
    rename = {}
    for c in df.columns:
        cc = str(c).strip().lower()
        if cc.startswith("solexs") or "sdd" in cc:
            rename[c] = "solexs_sdd2_counts"
        else:
            det = next((d for d in ("cdte1", "cdte2", "czt1", "czt2") if d in cc), None)
            if det is None:
                continue
            # find band bounds e.g. 5-20 or 5.0-20.0
            import re
            m = re.search(r"(\d+\.?\d*)\s*[-_]\s*(\d+\.?\d*)", cc)
            if m:
                low, high = float(m.group(1)), float(m.group(2))
                rename[c] = f"hel1os_{det}_{int(low)}-{int(high)}keV"
    return df.rename(columns=rename)


def prepare_feature_row(df: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Build features for the latest timestamp; return (features_df, last_raw)."""
    df = _map_columns(df.copy(), meta["feature_names"])
    have = [c for c in df.columns if c in set(meta["feature_names"])]
    if not have:
        raise ValueError(
            "No recognized channels found. Columns should look like: solexs_sdd2_counts, "
            "hel1os_cdte1_5-20keV, hel1os_czt2_40-60keV, ..."
        )
    feats = build_features(df[have])
    feats = feats.reindex(columns=meta["feature_names"])  # NaNs for missing
    feats = feats.fillna(pd.Series(meta["feature_medians"]))
    return feats, df


def predict_new_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Score the latest timestamp of new readings.

    Returns DataFrame with columns: time, flare_prob, alert.
    """
    meta = joblib.load(MODEL_PATH.replace(".joblib", "_meta.joblib"))
    model = joblib.load(MODEL_PATH)
    feats, raw = prepare_feature_row(df, meta)
    if len(feats) < 2:
        raise ValueError("Need at least a few rows to build the forecast.")
    last = feats.iloc[[-1]]
    proba = float(model.predict_proba(last)[:, 1][0])
    alert = "FLARE ALERT" if proba >= meta["threshold"] else "no flare expected"
    return pd.DataFrame({
        "time": [last.index[0]],
        "flare_prob": [proba],
        "alert": [alert],
        "threshold": [meta["threshold"]],
    })
