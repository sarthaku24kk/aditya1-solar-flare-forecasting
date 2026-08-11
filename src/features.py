"""Feature engineering: rolling statistics per channel + cross-channel ratios.

Every feature at time t uses only information available up to t (no lookahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CADENCE_S, LOOKBACK_WINDOWS_S, BACKGROUND_WINDOW_S


def _resample(df: pd.DataFrame) -> pd.DataFrame:
    """Downsample raw 1s data to CADENCE_S bins (mean)."""
    return df.resample(f"{CADENCE_S}s").mean()


def _rolling_features(s: pd.Series, win_s: int) -> pd.DataFrame:
    """Per-channel rolling statistics over a window of win_s seconds."""
    win = max(1, int(win_s / CADENCE_S))
    out = pd.DataFrame(index=s.index)
    base = f"w{win_s}"
    m = s.rolling(win, min_periods=1)
    out[f"{s.name}_mean_{base}"] = m.mean()
    out[f"{s.name}_max_{base}"] = m.max()
    out[f"{s.name}_std_{base}"] = m.std().fillna(0)
    # rate of change vs start of window
    out[f"{s.name}_slope_{base}"] = (s - s.shift(win)) / max(win, 1)
    return out


def _background_normalized(s: pd.Series) -> pd.Series:
    """flux / 2h-background (median), log-scaled."""
    bg = s.rolling(int(BACKGROUND_WINDOW_S / CADENCE_S), min_periods=1).median()
    return np.log1p(s) - np.log1p(bg)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from a (possibly raw 1s) channel DataFrame."""
    r = _resample(df)
    parts = [r.copy()]
    for col in r.columns:
        s = r[col].astype(float)
        parts.append(_background_normalized(s).to_frame(f"{col}_bglog"))
        parts.append(s.diff().clip(lower=0).to_frame(f"{col}_rise"))
        for w in LOOKBACK_WINDOWS_S:
            parts.append(_rolling_features(s, w))
    feats = pd.concat(parts, axis=1)
    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats


def sample_rows(index: pd.DatetimeIndex, every_s: int) -> np.ndarray:
    """Keep one index position every `every_s` seconds to decorrelate samples."""
    if len(index) == 0:
        return np.array([], dtype=int)
    step = max(1, int(every_s / CADENCE_S))
    return np.arange(0, len(index), step)
