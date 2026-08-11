"""Write sample upload templates for the demo app.

- quiet: a 3h quiet window (low probability expected)
- preflare: 2h window ending ~90 min before the big 2026-07-22 06:32 flare
  (model predicts onset in [t+60min, t+120min], so a high score is expected)
"""
from __future__ import annotations

import os

import pandas as pd

from src.data_loader import load

OUT = "output"
os.makedirs(OUT, exist_ok=True)
df = load()


def save(series_df: pd.DataFrame, name: str) -> None:
    out = series_df.reset_index().rename(columns={"index": "time"})
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(OUT, name)
    out.to_csv(path, index=False)
    print("saved", path, out.shape)


quiet = df.loc["2026-07-18 03:00":"2026-07-18 05:00"]
save(quiet, "sample_readings_quiet.csv")

preflare = df.loc["2026-07-22 03:00":"2026-07-22 05:00"]
save(preflare, "sample_readings_preflare.csv")
