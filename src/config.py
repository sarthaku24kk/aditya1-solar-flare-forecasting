"""Central configuration for the solar-flare forecast pipeline."""

# ---- data ----
DATA_PARQUET = "data/master.parquet"

# ---- flare detection (SoLEXS counts-based) ----
FLARE_WINDOW_S = 60        # smoothing window for background median
FLARE_MULT = 2.0           # flare = flux > mult * background
FLARE_MIN_PEAK = 120.0     # counts; min peak to count as a real flare
FLARE_MIN_DUR_S = 60.0     # min sustained duration above threshold

# ---- forecasting setup ----
CADENCE_S = 10             # feature grid cadence (seconds)
LOOKBACK_WINDOWS_S = [60, 300, 900, 1800, 3600]   # rolling windows used for features
BACKGROUND_WINDOW_S = 7200  # 2h background reference
HORIZON_S = 3600           # predict flare onset within the next 60 min
GAP_S = 3600               # minimum lead time: predict onset in [t+gap, t+gap+horizon]
SAMPLE_EVERY_S = 300       # keep one labelled sample every 5 min (avoids auto-correlation)
TRAIN_FRACTION = 0.7       # time-ordered split
WASHOUT_S = 7200           # gap between train end and test start (no feature leakage)

# ---- training ----
MODEL_PATH = "models/flare_model.joblib"
TEST_PLOTS = "output"

# ---- prediction thresholds (tune on validation) ----
THRESHOLD = 0.5
