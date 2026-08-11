"""Flare event detection and forecast-label construction.

Flare detection uses SoLEXS soft-X-ray total counts (background-subtracted
adaptive threshold). Onsets feed a binary forecast label: does a flare start
within the next HORIZON seconds?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (FLARE_WINDOW_S, FLARE_MULT, FLARE_MIN_PEAK,
                        FLARE_MIN_DUR_S)


def detect_flare_events(flux: pd.Series, dt_s: float = 1.0) -> pd.DataFrame:
    """Find flare events in a 1-second flux series.

    Returns DataFrame with columns: onset, peak_time, peak_flux, end, duration_s.
    """
    flux = flux.astype(float)
    background = flux.rolling(int(FLARE_WINDOW_S), min_periods=1).median()
    ratio = flux / background.replace(0, np.nan)
    # only require the ratio threshold when background is non-trivial
    active = (flux >= FLARE_MULT * background) | (flux >= FLARE_MIN_PEAK)
    active = active.fillna(False)

    diff = active.astype(int).diff().fillna(0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    events = []
    for s in starts:
        e = ends[ends > s][0] if np.any(ends > s) else len(flux) - 1
        seg = flux.iloc[s:e + 1]
        peak_i = seg.idxmax()
        if seg.max() < FLARE_MIN_PEAK:
            continue
        duration = len(seg) * dt_s
        if duration < FLARE_MIN_DUR_S:
            continue
        events.append({
            "onset": seg.index[0],
            "peak_time": peak_i,
            "peak_flux": float(seg.max()),
            "end": seg.index[-1],
            "duration_s": duration,
        })
    return pd.DataFrame(events)


def build_labels(index: pd.DatetimeIndex, onsets: list, horizon_s: int,
                 gap_s: int = 0) -> pd.Series:
    """Binary label: 1 if a flare onset falls within (t+gap, t+gap+horizon].

    gap_s enforces a minimum lead time so the model must predict from
    quiet pre-flare data, not while the flare is already rising.
    """
    onset_arr = np.asarray([t.value for t in pd.to_datetime(onsets)])
    idx_ns = index.values.astype("datetime64[ns]").astype(np.int64)
    gap_ns = int(gap_s) * 1_000_000_000
    horizon_ns = int(horizon_s) * 1_000_000_000
    labels = np.zeros(len(index), dtype=np.int8)
    for o in onset_arr:
        after = idx_ns >= o + gap_ns
        within = idx_ns <= o + gap_ns + horizon_ns
        labels[after & within] = 1
    return pd.Series(labels, index=index)
