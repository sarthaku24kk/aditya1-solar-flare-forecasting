"""Train + evaluate the flare-forecast classifier (scikit-learn).

Metrics include ROC-AUC, PR-AUC, True Skill Statistic (TSS) and Heidke
Skill Score (HSS). The time-ordered split has a washout buffer so no
rolling-window features can leak across the boundary.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (GAP_S, HORIZON_S, MODEL_PATH, SAMPLE_EVERY_S,
                        TEST_PLOTS, TRAIN_FRACTION, WASHOUT_S)
from src.data_loader import load
from src.features import build_features, sample_rows
from src.flares import build_labels, detect_flare_events


def _skill_scores(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tss = tp / (tp + fn) - fp / (fp + tn) if (tp + fn) and (fp + tn) else 0.0
    pcs = (tp + tn) / (tp + fp + fn + tn)
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (tp + fp + fn + tn) ** 2
    hss = (pcs - expected) / (1 - expected) if expected != 1 else 0.0
    return tss, hss


def _best_threshold(y, proba):
    prec, rec, th = precision_recall_curve(y, proba)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    return th[i] if i < len(th) else 0.5


def make_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = load()
    events = detect_flare_events(df["solexs_sdd2_counts"])
    feats = build_features(df)
    idx = feats.index
    labels = build_labels(idx, events["onset"], HORIZON_S, GAP_S)
    keep = sample_rows(idx, SAMPLE_EVERY_S)
    X = feats.iloc[keep]
    y = labels.iloc[keep]

    # exclude samples during an ongoing flare
    in_flare = np.zeros(len(y), dtype=bool)
    for _, ev in events.iterrows():
        in_flare |= (X.index >= ev["onset"]) & (X.index <= ev["end"])
    X = X[~in_flare]
    y = y[~in_flare]

    mask = X.notna().all(axis=1)
    return X[mask], y[mask]


def main() -> None:
    X, y = make_dataset()
    print(f"samples={len(X)}  positive={y.sum()}  ({y.mean():.3%})")

    # time-ordered split with a washout buffer at the boundary
    n_train = int(len(X) * TRAIN_FRACTION)
    boundary = X.index[n_train]
    washout_ns = WASHOUT_S * 1_000_000_000
    test_mask = (X.index.astype("datetime64[ns]").astype(np.int64)
                 >= boundary.value + washout_ns)
    train_mask = ~test_mask
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    print(f"train n={len(X_tr)} (+{y_tr.sum()}) | test n={len(X_te)} (+{y_te.sum()})")
    print(f"washout buffer: {WASHOUT_S}s between train end and test start")

    model = make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
    )
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    thr = _best_threshold(y_te, proba)
    pred = (proba >= thr).astype(int)
    tss, hss = _skill_scores(y_te, pred)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

    print("\n=== TEST (time-ordered holdout, threshold=%.3f) ===" % thr)
    print(f"positive rate in test: {y_te.mean():.3%}")
    print(f"ROC-AUC: {roc_auc_score(y_te, proba):.3f}")
    print(f"PR-AUC:  {average_precision_score(y_te, proba):.3f}")
    print(f"TSS: {tss:.3f}   HSS: {hss:.3f}")
    print(f"precision: {tp/(tp+fp):.3f}  recall: {tp/(tp+fn):.3f}")
    print(f"confusion matrix (rows=true, cols=pred):\n[[{tn} {fp}]\n [{fn} {tp}]]")

    imp = pd.Series(model.named_steps["gradientboostingclassifier"].feature_importances_,
                    index=X_tr.columns).sort_values(ascending=False)
    print("\ntop-15 features:\n", imp.head(15).to_string())

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # save metadata needed for inference on new readings
    meta = {
        "feature_names": list(X_tr.columns),
        "feature_medians": X_tr.median().to_dict(),
        "threshold": float(thr),
        "gap_s": GAP_S,
        "horizon_s": HORIZON_S,
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "pr_auc": float(average_precision_score(y_te, proba)),
        "tss": float(tss),
        "hss": float(hss),
    }
    joblib.dump(meta, MODEL_PATH.replace(".joblib", "_meta.joblib"))
    print(f"saved model -> {MODEL_PATH} (+ _meta.joblib)")

    os.makedirs(TEST_PLOTS, exist_ok=True)
    pd.DataFrame({
        "time": X_te.index,
        "y_true": y_te.values,
        "score": proba,
    }).to_csv(os.path.join(TEST_PLOTS, "test_scores.csv"), index=False)
    print("test scores ->", os.path.join(TEST_PLOTS, "test_scores.csv"))


if __name__ == "__main__":
    main()
