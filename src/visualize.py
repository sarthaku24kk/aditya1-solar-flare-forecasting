"""Diagnostic plots: full-time prediction vs flare events, PR curve, feature importance."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import TEST_PLOTS
from src.data_loader import load
from src.features import build_features
from src.flares import detect_flare_events
from src.predict import forecast

os.makedirs(TEST_PLOTS, exist_ok=True)


def plot_timeline() -> None:
    scores = forecast()
    df = load()
    events = detect_flare_events(df["solexs_sdd2_counts"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(df.index, df["solexs_sdd2_counts"], lw=0.4, color="tab:red")
    axes[0].set_ylabel("SoLEXS counts/s")
    axes[0].set_title("SoLEXS soft X-ray + detected flare peaks")
    for _, ev in events.iterrows():
        axes[0].axvline(ev["onset"], color="k", ls="--", lw=0.6, alpha=0.5)
    axes[0].scatter(events["peak_time"], events["peak_flux"], s=25, zorder=5, color="black")

    axes[1].plot(df.index, df["hel1os_czt2_40-60keV"], lw=0.4, color="tab:blue")
    axes[1].set_ylabel("HEL1OS 40-60 keV cts/s")
    axes[1].set_title("HEL1OS hard X-ray")

    axes[2].plot(scores.index, scores.values, lw=0.8, color="tab:green")
    axes[2].set_ylabel("P(flare within next 60 min)")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Model forecast probability")
    for _, ev in events.iterrows():
        axes[2].axvline(ev["onset"], color="k", ls="--", lw=0.6, alpha=0.5)

    plt.tight_layout()
    out = os.path.join(TEST_PLOTS, "timeline.png")
    plt.savefig(out, dpi=110)
    plt.close()
    print("saved", out)


def plot_test_metrics() -> None:
    s = pd.read_csv(os.path.join(TEST_PLOTS, "test_scores.csv"))
    from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score

    y, p = s["y_true"].values, s["score"].values
    prec, rec, _ = precision_recall_curve(y, p)
    fpr, tpr, _ = roc_curve(y, p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(rec, prec, color="tab:blue")
    axes[0].axhline(y.mean(), color="gray", ls="--", label="baseline (pos rate)")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Precision-Recall (PR-AUC={average_precision_score(y, p):.3f})")
    axes[0].legend()
    from sklearn.metrics import roc_auc_score
    axes[1].plot(fpr, tpr, color="tab:green")
    axes[1].plot([0, 1], [0, 1], ls="--", color="gray")
    axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
    axes[1].set_title(f"ROC (AUC={roc_auc_score(y, p):.3f})")
    plt.tight_layout()
    out = os.path.join(TEST_PLOTS, "metrics.png")
    plt.savefig(out, dpi=110)
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    plot_timeline()
    plot_test_metrics()
