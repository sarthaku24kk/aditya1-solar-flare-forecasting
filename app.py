"""Streamlit demo: solar-flare forecast from SoLEXS + HEL1OS readings.

Run:  streamlit run app.py
"""
from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH
from src.data_loader import load
from src.flares import detect_flare_events
from src.ingest import load_uploaded, load_uploaded_zip, predict_new_readings

st.set_page_config(page_title="Aditya-L1 Flare Forecast", layout="wide")

MODEL_META = joblib.load(MODEL_PATH.replace(".joblib", "_meta.joblib"))

st.title("Solar Flare Forecast - Aditya-L1 (SoLEXS + HEL1OS)")
st.caption(
    "Predicts the probability of a solar flare starting in the next "
    f"window (min lead {MODEL_META['gap_s']/60:.0f} min, horizon "
    f"{MODEL_META['horizon_s']/60:.0f} min). Model: GradientBoosting, "
    f"holdout ROC-AUC {MODEL_META['roc_auc']:.3f}."
)

tab1, tab2 = st.tabs(["Demo mode (real data)", "Upload your readings"])


# ---------------------------------------------------------------- demo mode
with tab1:
    st.subheader("Replay from real Aditya-L1 data")
    df = load()
    demo_time = st.slider(
        "Simulate 'now' (UTC)",
        min_value=df.index[0].to_pydatetime() + pd.Timedelta(hours=2),
        max_value=df.index[-1].to_pydatetime(),
        value=df.index[-1].to_pydatetime(),
        step=pd.Timedelta(minutes=5),
    )
    as_of = pd.Timestamp(demo_time)

    from src.ingest import prepare_feature_row

    meta = MODEL_META
    window = df.loc[as_of - pd.Timedelta(hours=3): as_of]
    if len(window) < 600:
        st.warning("Not enough history in the selected window.")
    else:
        feats, _ = prepare_feature_row(window, meta)
        model = joblib.load(MODEL_PATH)
        proba = float(model.predict_proba(feats.iloc[[-1]])[:, 1][0])

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted flare probability", f"{proba:.1%}")
        c2.metric("Alert threshold", f"{meta['threshold']:.1%}")
        alarm = "RED - FLARE ALERT" if proba >= meta["threshold"] else "GREEN - quiet"
        c3.metric("Status", alarm)

        st.line_chart(window["solexs_sdd2_counts"].rename("SoLEXS counts/s"))
        st.line_chart(window["hel1os_czt2_40-60keV"].rename("HEL1OS 40-60 keV"))
        st.write(f"Forecast as of **{as_of}**")


# ------------------------------------------------------------ upload mode
with tab2:
    st.subheader("Upload new SoLEXS/HEL1OS readings")
    st.markdown(
        "CSV/XLSX with a `time` column and channel columns, **or a zip of "
        "Aditya-L1 FITS products** (SoLEXS `.pi` + HEL1OS `lightcurve_*.fits`, "
        "plain or `.gz` — nested folders OK; the app converts it automatically). "
        "Accepted channel names: `solexs_sdd2_counts`, `hel1os_cdte1_5-20keV`, "
        "`hel1os_cdte2_40-60keV`, `hel1os_czt2_18-160keV`, ... "
        "(detector + band are auto-detected)."
    )
    uploaded = st.file_uploader(
        "Choose file", type=["csv", "xlsx", "xls", "zip"]
    )
    if uploaded is not None:
        try:
            name = uploaded.name.lower()
            if name.endswith(".zip"):
                raw = load_uploaded_zip(uploaded.getvalue(), uploaded.name)
                st.success(
                    "Zip parsed automatically: SoLEXS spectra + HEL1OS "
                    "light curves extracted and converted to channel columns."
                )
            else:
                raw = load_uploaded(uploaded.getvalue(), uploaded.name)
            st.write(f"Loaded {len(raw)} rows, {len(raw.columns)} channels "
                     f"({raw.index.min()} to {raw.index.max()}).")
            st.dataframe(raw.head(10), use_container_width=True)
            result = predict_new_readings(raw)
            proba = float(result["flare_prob"][0])
            st.metric("Predicted flare probability", f"{proba:.1%}")
            st.metric("Status", str(result["alert"][0]))
            st.line_chart(raw)
        except Exception as e:  # noqa: BLE001 - surface all errors to the demo
            st.error(f"Error: {e}")

st.divider()
st.caption(
    "Trained on 20 days of Aditya-L1 data (2026-07-18 to 08-06). "
    "Flare events are auto-detected from SoLEXS soft X-ray counts."
)
