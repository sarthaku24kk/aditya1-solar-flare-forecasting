"""Real-time nowcast/forecast: score the latest cadence using a trained model."""
from __future__ import annotations

import joblib
import pandas as pd

from src.config import HORIZON_S, MODEL_PATH
from src.features import build_features
from src.flares import build_labels, detect_flare_events
from src.data_loader import load


def forecast(model_path=MODEL_PATH, as_of=None, horizon_s=HORIZON_S):
    """Return a DataFrame of (time, flare_probability) for each sampled time.

    The probability means: "a flare will START within the next horizon_s".
    """
    model = joblib.load(model_path)
    df = load()
    feats = build_features(df)
    idx = feats.index
    X = feats.loc[idx[feats.notna().all(axis=1)]]
    scores = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="flare_prob")
    if as_of is not None:
        scores = scores.loc[:pd.Timestamp(as_of)]
    return scores


if __name__ == "__main__":
    scores = forecast()
    print(scores.tail(20))
    print("\nlatest:", scores.index[-1],
          "P(flare within next hr) =", round(float(scores.iloc[-1]), 4))
