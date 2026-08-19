"""Load SoLEXS (.pi spectra) and HEL1OS (lightcurve fits) into one time-indexed DataFrame.

Caches the result as data/master.parquet for fast reloads.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLEXS_DIR = os.path.join(ROOT, "data", "solexs")
HEL1OS_DIR = os.path.join(ROOT, "data", "hel1os")
CACHE = os.path.join(ROOT, "data", "master.parquet")

UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _to_utc(tstart: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(tstart, dtype=np.float64), unit="s", utc=True)


def load_solexs(solexs_dir: str | None = None) -> pd.DataFrame:
    """Sum SoLEXS SDD2 spectra over channels -> per-second total counts."""
    solexs_dir = solexs_dir or SOLEXS_DIR
    frames = []
    for f in sorted(glob.glob(os.path.join(solexs_dir, "**", "*.pi"), recursive=True)):
        if "SDD1" in f:  # SDD1 has no data (empty GTI)
            continue
        with fits.open(f) as h:
            d = h[1].data
        t = np.asarray(d["TSTART"], dtype=np.float64)
        counts = np.nansum(np.asarray(d["COUNTS"], dtype=np.float64), axis=1)
        df = pd.DataFrame({"solexs_sdd2_counts": counts}, index=_to_utc(t))
        df.index = df.index.tz_localize(None)
        frames.append(df)
    out = pd.concat(frames, axis=0)
    if out.empty:
        return out
    return out[~out.index.duplicated(keep="last")].sort_index()


def load_hel1os(hel1os_dir: str | None = None) -> pd.DataFrame:
    """Read every HEL1OS light-curve energy band as one column.

    Column names: hel1os_<det>_<elow>-<ehigh>keV  e.g. hel1os_cdte1_5-20keV
    """
    hel1os_dir = hel1os_dir or HEL1OS_DIR
    per_det: dict[str, list[pd.DataFrame]] = {}
    for f in sorted(glob.glob(os.path.join(hel1os_dir, "**", "lightcurve_*.fits"), recursive=True)):
        det = os.path.basename(f).replace("lightcurve_", "").replace(".fits", "")
        with fits.open(f) as h:
            file_cols = {}
            for hdu in h:
                if hdu.data is None or hdu.name == "PRIMARY":
                    continue
                d = hdu.data
                low, high = hdu.header.get("ELOW", 0), hdu.header.get("EHIGH", 0)
                col = f"hel1os_{det}_{int(low)}-{int(high)}keV"
                mjd = np.asarray(d["MJD"], dtype=np.float64)
                secs = np.rint((mjd - 55197) * 86400.0).astype(np.int64)  # 55197 = MJD of 2010-01-01
                ts = pd.to_datetime("2010-01-01") + pd.to_timedelta(secs, unit="s")
                s = pd.Series(np.asarray(d["CTR"], dtype=np.float64), index=ts, name=col)
                file_cols[col] = s
            df = pd.DataFrame(file_cols)
            per_det.setdefault(det, []).append(df)
    det_frames = []
    for det, frames in per_det.items():
        stacked = pd.concat(frames, axis=0)
        stacked = stacked[~stacked.index.duplicated(keep="last")].sort_index()
        det_frames.append(stacked)
    if not det_frames:
        return pd.DataFrame()
    out = pd.concat(det_frames, axis=1)
    return out.sort_index()


def load(force: bool = False, cache: bool = True) -> pd.DataFrame:
    """Unified 1-second cadence DataFrame of SoLEXS + HEL1OS channels."""
    if cache and os.path.exists(CACHE) and not force:
        return pd.read_parquet(CACHE)
    solexs = load_solexs()
    hel1os = load_hel1os()
    out = solexs.join(hel1os, how="outer").sort_index()
    if cache:
        out.to_parquet(CACHE)
    return out


if __name__ == "__main__":
    df = load()
    print(df.shape)
    print(df.index.min(), "->", df.index.max())
    print(df.notna().mean().round(3))
