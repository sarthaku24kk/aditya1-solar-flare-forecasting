"""Turn new raw readings (uploaded CSV) into model-ready features and a forecast.

The input CSV must have a time column and at least one channel column.
Column mapping is fuzzy: solexs* -> SoLEXS channel, hel1os_<det>_<elow>-<ehigh>keV
channels are matched by detector name. Missing channels are filled with the
training-set median (neutral value) so predictions still work.
"""
from __future__ import annotations

import gzip
import io
import os
import shutil
import tempfile
import zipfile

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
    """Parse an uploaded file (CSV/XLSX) into a DataFrame with a datetime index."""
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


def load_uploaded_zip(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse a zip of Aditya-L1 FITS products (.pi spectra + light curves).

    Files may be plain FITS or gzipped (.gz); nested folders are handled.
    """
    from src.data_loader import load_hel1os, load_solexs

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            for m in zf.infolist():
                if m.is_dir():
                    continue
                data = zf.read(m)
                name = m.filename
                if name.endswith(".gz"):
                    data = gzip.decompress(data)
                    name = name[:-3]
                base = name.split("/")[-1].lower()
                if not (base.endswith(".pi") or base.startswith("lightcurve_")):
                    continue
                target = os.path.join(tmp, name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as out:
                    out.write(data)

        solexs = load_solexs(tmp)
        hel1os = load_hel1os(tmp)
        if solexs.empty and hel1os.empty:
            raise ValueError(
                "No SoLEXS (.pi) or HEL1OS (lightcurve_*.fits) files found in the zip."
            )
        out = solexs.join(hel1os, how="outer").sort_index()
    return out


def _map_columns(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Rename uploaded columns to the canonical feature-channel names."""
    rename = {}
    for c in df.columns:
        cc = str(c).strip().lower()
        if cc.startswith("solexs") or "sdd" in cc:
            rename[c] = "solexs_sdd2_counts"
        else:
            import re
            det = next((d for d in ("cdte1", "cdte2", "czt1", "czt2") if d in cc), None)
            if det is None:
                continue
            # band bounds like 5-20keV / 5.0-20.0keV (match keV-anchored pair,
            # avoiding the detector number inside 'cdte1_5-20keV')
            m = re.search(r"(\d+(?:\.\d+)?)\s*[-_]\s*(\d+(?:\.\d+)?)kev", cc)
            if m:
                low, high = float(m.group(1)), float(m.group(2))
                rename[c] = f"hel1os_{det}_{int(low)}-{int(high)}keV"
            else:
                m = re.search(r"[_-](\d+(?:\.\d+)?)[-_](\d+(?:\.\d+)?)\b", cc)
                if m:
                    low, high = float(m.group(1)), float(m.group(2))
                    rename[c] = f"hel1os_{det}_{int(low)}-{int(high)}keV"
    return df.rename(columns=rename)


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge parsed uploads onto a single time axis.

    Same-channel files (e.g. two 12h HEL1OS halves) are stacked on rows;
    different instruments (SoLEXS vs HEL1OS) are joined on the index.
    """
    solexs, hel1os = [], []
    for df in frames:
        if "solexs_sdd2_counts" in df.columns:
            solexs.append(df)
        hel = [c for c in df.columns if c.startswith("hel1os_")]
        if hel:
            hel1os.append(df[hel])
    parts = []
    if solexs:
        s = pd.concat(solexs, axis=0)
        parts.append(s[~s.index.duplicated(keep="last")].sort_index())
    if hel1os:
        per_det: dict[str, list[pd.DataFrame]] = {}
        for df in hel1os:
            for det in ("cdte1", "cdte2", "czt1", "czt2"):
                cols = [c for c in df.columns if f"hel1os_{det}_" in c]
                if cols:
                    per_det.setdefault(det, []).append(df[cols])
        det_frames = []
        for det, chunk in per_det.items():
            stacked = pd.concat(chunk, axis=0)
            stacked = stacked[~stacked.index.duplicated(keep="last")].sort_index()
            det_frames.append(stacked)
        parts.append(pd.concat(det_frames, axis=1).sort_index())
    if not parts:
        return pd.DataFrame()
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, how="outer").sort_index()
    return out


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
