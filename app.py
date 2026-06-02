"""Streamlit web app for the Child Mind Institute - Detect Sleep States project.

Loads a trained scikit-learn pipeline and detects sleep ``onset`` / ``wakeup``
events from wrist-worn accelerometer data (``enmo`` and ``anglez``, every 5 s).

The app is organised into tabs:

* **Analyze**  - single-series KPIs, interactive chart, per-night breakdown,
  clinical sleep metrics and a downloadable report.
* **Batch**    - summary table across every series in the data.
* **Compare**  - two series side by side.
* **Model**    - model card with validation metrics and feature importances.

Run with::

    streamlit run app.py
"""

import hashlib
import inspect
import json
import os
import pickle

import pandas as pd
import streamlit as st
from PIL import Image

from utils import (
    MIN_SLEEP_STEPS,
    SMOOTH_WINDOW,
    STEPS_PER_MINUTE,
    batch_summary,
    build_html_report,
    build_submission,
    check_df,
    compute_sleep_metrics,
    feature_engineering,
    generate_synthetic_series,
    get_predictions_probas,
    get_series_ids,
    per_night_breakdown,
    plotly_actogram,
    plotly_feature_importance,
    plotly_nightly_trend,
    plotly_prediction,
    rate_metric,
    read_file,
    recording_span,
    set_chart_theme,
)

# --- Bundled assets ----------------------------------------------------------
# Prefer the modern, self-contained pipeline_v2.pkl (loads on any scikit-learn);
# fall back to the legacy pickle (scikit-learn 1.3.x only) if v2 is missing.
PIPELINE_CANDIDATES = ["pipeline_v2.pkl", "pipeline_01.pkl", "pipeline_01_legacy.pkl"]
DATA_DIR = "data"
METRICS_PATH = os.path.join("models", "model_metrics.json")
SAMPLE_SERIES = {
    "Sample 1": os.path.join(DATA_DIR, "input_series_1.csv"),
    "Sample 2": os.path.join(DATA_DIR, "input_series_2.csv"),
    "Sample 3": os.path.join(DATA_DIR, "input_series_3.csv"),
}

st.set_page_config(
    page_title="Detect Sleep States",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme: default dark; the sidebar toggle flips this and reruns. We apply it
# early so the CSS and Plotly charts pick up the right palette.
dark_mode = st.session_state.get("dark_mode", True)
set_chart_theme(dark=dark_mode)

# Shared styling (hero, metric cards, download button) plus a light-mode page
# override (the base config.toml theme is dark).
_light_override = (
    ""
    if dark_mode
    else """
      .stApp { background-color: #f7f7fb; }
      .stApp, .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2,
      .stApp h3, .stApp h4 { color: #1a1a2e; }
      [data-testid="stSidebar"] { background-color: #ececf5; }
      div[data-testid="stMetric"] { background: rgba(109, 40, 217, 0.06); }
    """
)
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.0rem; }}
      div[data-testid="stMetric"] {{
        background: rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 12px; padding: 14px 16px;
      }}
      div[data-testid="stMetricLabel"] {{ opacity: 0.8; }}
      .hero {{
        background: linear-gradient(120deg, #1e1b4b 0%, #4c1d95 55%, #6d28d9 100%);
        border-radius: 16px; padding: 24px 30px; margin-bottom: 10px;
      }}
      .hero h1 {{ margin: 0; font-size: 1.9rem; color: #fff; }}
      .hero p  {{ margin: 6px 0 0 0; opacity: 0.9; color: #fff; }}
      .stDownloadButton button, div[data-testid="stDownloadButton"] button {{
        background: #6d28d9; color: #fff; border: 0; border-radius: 10px;
      }}
      {_light_override}
    </style>
    """,
    unsafe_allow_html=True,
)


# `key=` was added to st.plotly_chart in Streamlit 1.30. On 1.30+ (incl. the
# version Streamlit Cloud installs) a unique key prevents
# StreamlitDuplicateElementId when two charts share parameters; on older
# Streamlit (e.g. 1.28) the kwarg is unsupported, so we pass it only when
# available.
_PLOTLY_SUPPORTS_KEY = "key" in inspect.signature(st.plotly_chart).parameters


def plot(fig, *, key: str):
    """st.plotly_chart with a unique key, compatible across Streamlit versions."""
    if _PLOTLY_SUPPORTS_KEY:
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True)


# --- Cached compute ----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load and cache the trained pipeline. Returns ``(pipeline, path)``."""
    for path in PIPELINE_CANDIDATES:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f), path
    raise FileNotFoundError("No model file found: " + ", ".join(PIPELINE_CANDIDATES))


@st.cache_data(show_spinner=False)
def load_sample(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of a dataframe's content for cache keying."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


@st.cache_data(show_spinner=False)
def _engineer_cached(_df: pd.DataFrame, fp: str) -> pd.DataFrame:
    """Feature-engineer once per unique dataframe (keyed on its fingerprint)."""
    return feature_engineering(_df)


@st.cache_data(show_spinner=False)
def _predict_cached(_prepared: pd.DataFrame, fp: str, threshold: float):
    """Predictions cached per (data, threshold)."""
    pipeline, _ = load_pipeline()
    return get_predictions_probas(_prepared, pipeline, threshold=threshold)


def analyze(df: pd.DataFrame, threshold: float, smooth_window: int, min_sleep_steps: int):
    """Run the full flow once and return a context dict (or an error message)."""
    fp = _fingerprint(df)
    try:
        prepared = _engineer_cached(df, fp)
        y_pred, y_probas = _predict_cached(prepared, fp, threshold)
    except AttributeError:
        return {"error": "model_version"}
    try:
        submission = build_submission(
            prepared, y_pred, y_probas, smooth_window, min_sleep_steps
        )
    except (ValueError, IndexError):
        return {"error": "no_periods", "prepared": prepared}
    return {"prepared": prepared, "submission": submission}


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    head_l, head_r = st.columns([3, 2])
    head_l.markdown("### 🌙 Sleep States")
    # Theme toggle (checkbox works on Streamlit 1.28; st.toggle does not).
    head_r.checkbox("🌗 Dark", value=dark_mode, key="dark_mode")
    st.caption("Detect sleep onset & wakeup from accelerometer data.")
    st.divider()

    st.markdown("#### 1 · Choose your data")
    source = st.radio(
        "Data source",
        ["Use a sample series", "Upload my own data", "Generate synthetic data"],
        label_visibility="collapsed",
    )

    active_df: pd.DataFrame | None = None
    data_label = ""

    if source == "Upload my own data":
        uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
        if uploaded is not None:
            result = read_file(uploaded.name, uploaded)
            if isinstance(result, str):
                st.error(result)
            else:
                active_df, data_label = result, uploaded.name
    elif source == "Generate synthetic data":
        n_series = st.slider("Number of series", 1, 5, 2)
        days = st.slider("Days per series", 2, 7, 4)
        if st.button("✨ Generate", use_container_width=True):
            st.session_state["synthetic"] = generate_synthetic_series(
                n_series=n_series, days=days, seed=7
            )
        if "synthetic" in st.session_state:
            active_df = st.session_state["synthetic"]
            data_label = f"synthetic ({n_series} series)"
            st.download_button(
                "⬇️ Download generated CSV",
                active_df.to_csv(index=True).encode("utf-8"),
                file_name="synthetic_series.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        choice = st.selectbox("Sample series", list(SAMPLE_SERIES.keys()))
        active_df = load_sample(SAMPLE_SERIES[choice])
        data_label = choice

    # Validate.
    if active_df is not None:
        check = check_df(active_df)
        if check is not True:
            st.error("The data is not valid:")
            for elem in check:
                st.write(" - ", elem)
            active_df = None

    # Inline data preview so users can confirm the file parsed correctly.
    if active_df is not None:
        with st.expander(f"👁️ Preview ({len(active_df):,} rows)"):
            st.dataframe(active_df.head(8), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 2 · Detection settings")

    # Default slider values (also used by the reset button below).
    DEFAULTS = {
        "threshold": 0.50,
        "smooth_min": SMOOTH_WINDOW // STEPS_PER_MINUTE,
        "min_sleep_min": MIN_SLEEP_STEPS // STEPS_PER_MINUTE,
    }
    if st.button("↺ Reset to defaults", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    threshold = st.slider(
        "Sleep/wake threshold", 0.05, 0.95, DEFAULTS["threshold"], 0.05,
        key="threshold",
        help="Higher = the model must be more confident the child is awake.",
    )
    smooth_min = st.slider(
        "Smoothing window (minutes)", 5, 180, DEFAULTS["smooth_min"], 5,
        key="smooth_min",
        help="Rolling window used to remove short flickers before detecting events.",
    )
    min_sleep_min = st.slider(
        "Minimum sleep period (minutes)", 15, 240, DEFAULTS["min_sleep_min"], 15,
        key="min_sleep_min",
    )
    smooth_window = smooth_min * STEPS_PER_MINUTE
    min_sleep_steps = min_sleep_min * STEPS_PER_MINUTE

    st.divider()
    _, model_path = (None, "—")
    try:
        _, model_path = load_pipeline()
    except Exception:
        pass
    st.caption(f"Model: `{os.path.basename(model_path)}`")
    st.caption(
        "[Kaggle competition](https://www.kaggle.com/competitions/"
        "child-mind-institute-detect-sleep-states) · "
        "[Child Mind Institute](https://childmind.org/)"
    )


# --- Hero --------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>😴 Child Mind Institute — Detect Sleep States</h1>
      <p>Detection of sleep onset and wake from wrist-worn accelerometer data ⌚</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _model_version_error():
    import sklearn

    st.error(
        f"❌ The model could not run with scikit-learn {sklearn.__version__}. "
        "Re-train a portable model with `python scripts/train_model.py`, or run "
        "in the bundled Python 3.11 environment (`.venv311`)."
    )


# Compute the analysis ONCE (cached) before the tabs, with a visible status so
# the main area never sits blank while the model runs.
ctx: dict | None = None
if active_df is not None:
    n_rows = len(active_df)
    with st.status(
        f"Analysing {n_rows:,} readings · {active_df['series_id'].nunique()} series…",
        expanded=False,
    ) as status:
        st.write("Engineering features…")
        ctx = analyze(active_df, threshold, smooth_window, min_sleep_steps)
        if ctx.get("error") == "model_version":
            status.update(label="Model could not load", state="error")
        elif ctx.get("error") == "no_periods":
            status.update(label="No sleep periods found with current settings", state="error")
        else:
            status.update(label="Analysis complete ✓", state="complete")

# --- Tabs --------------------------------------------------------------------
tab_analyze, tab_batch, tab_compare, tab_model = st.tabs(
    ["🔍 Analyze", "📊 Batch", "🆚 Compare", "🤖 Model"]
)

if active_df is None:
    with tab_analyze:
        st.info("👈 Pick a **sample**, **upload** data, or **generate synthetic** data in the sidebar.")
        st.markdown(
            """
            #### How it works
            The device records two signals every **5 seconds** — **enmo** (movement
            magnitude) and **anglez** (arm angle). The model engineers features and
            predicts when the child **falls asleep** (onset) and **wakes up** (wakeup).
            Use the **detection settings** in the sidebar to tune sensitivity.
            """
        )

# ---- Analyze tab ----
with tab_analyze:
    if active_df is not None:
        series_ids = get_series_ids(active_df)
        sid = st.selectbox("Series to view", series_ids, key="analyze_series")
        show_anglez = st.checkbox("Show anglez trace", value=False, key="show_anglez")
        span = recording_span(active_df, sid)
        st.caption(f"Showing **{data_label}** · series `{sid}` · 📅 {span['label']}")

        if ctx.get("error") == "model_version":
            _model_version_error()
        elif ctx.get("error") == "no_periods":
            st.warning(
                "⚠️ No complete sleep period (onset → wakeup) longer than the minimum "
                "was found for this series. Try lowering the **minimum sleep period** "
                "or the **threshold** in the sidebar."
            )
        else:
            prepared, submission = ctx["prepared"], ctx["submission"]
            m = compute_sleep_metrics(active_df, submission, sid)

            eff_label, eff_color = rate_metric("efficiency", m["efficiency"])
            lat_label, lat_color = rate_metric("latency", m["latency_h"])
            awk_label, awk_color = rate_metric("awakenings", m["awakenings"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("😴 Sleep periods", m["periods"])
            c2.metric("⏱️ Total sleep", m["total_sleep"])
            c3.metric("🛏️ Efficiency", f"{m['efficiency']}%",
                      delta=eff_label, delta_color=eff_color,
                      help="Share of the recording's span spent asleep. "
                           "Reference: ≥85% good, 75–85% fair, <75% low.")
            c4.metric("🎯 Avg confidence", f"{m['avg_confidence'] * 100:.1f}%")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("🌙 Sleep latency",
                      f"{m['latency_h']}h" if m["latency_h"] is not None else "—",
                      delta=lat_label, delta_color=lat_color,
                      help="Time from recording start to the first sleep onset.")
            c6.metric("➿ Awakenings", m["awakenings"],
                      delta=awk_label, delta_color=awk_color)
            c7.metric("📈 Longest", f"{m['longest_h']}h")
            c8.metric("📉 Shortest", f"{m['shortest_h']}h")

            plot(
                plotly_prediction(prepared, submission, sid, show_anglez=show_anglez),
                key=f"timeline_{sid}",
            )

            per_night = per_night_breakdown(active_df, submission, sid)
            left, right = st.columns([3, 2])
            with left:
                st.markdown("**Per-night breakdown**")
                # Static st.table (formatted) avoids the Streamlit 1.28 data-grid
                # "Row index is out of range" console error on re-render.
                pn_display = per_night.copy()
                for c in ("onset", "wakeup"):
                    pn_display[c] = pd.to_datetime(pn_display[c]).dt.strftime("%b %d, %H:%M")
                pn_display["duration_h"] = pn_display["duration_h"].map(lambda v: f"{v:.2f} h")
                pn_display["confidence"] = pn_display["confidence"].map(lambda v: f"{v:.2f}")
                st.table(pn_display.set_index("night"))
            with right:
                plot(plotly_nightly_trend(per_night), key=f"trend_{sid}")

            actogram = plotly_actogram(prepared, submission, sid)
            if actogram is not None:
                plot(actogram, key=f"actogram_{sid}")

            with st.expander("Predicted events table"):
                st.dataframe(
                    submission.drop(columns=["row_id"]),
                    use_container_width=True, hide_index=True,
                )

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    label="⬇️ Download predictions (CSV)",
                    data=submission.to_csv(index=False).encode("utf-8"),
                    file_name=f"predictions_{sid}.csv",
                    mime="text/csv",
                    key=f"dl_pred_{sid}",
                    use_container_width=True,
                )
            with d2:
                st.download_button(
                    "📄 Download report (HTML)",
                    data=build_html_report(sid, m, per_night).encode("utf-8"),
                    file_name=f"sleep_report_{sid}.html",
                    mime="text/html",
                    use_container_width=True,
                )

# ---- Batch tab ----
with tab_batch:
    if active_df is None:
        st.info(
            "📊 **Batch dashboard** — load a multi-series file to see a summary "
            "row per series (total sleep, efficiency, confidence).\n\n"
            "👈 Tip: use **Generate synthetic data** with 2+ series to try it."
        )
    else:
        st.markdown("#### Summary across all series")
        if ctx.get("error") == "model_version":
            _model_version_error()
        elif "submission" not in ctx:
            st.warning("No sleep periods could be derived for any series with the current settings.")
        else:
            table = batch_summary(active_df, ctx["submission"])
            t1, t2, t3 = st.columns(3)
            t1.metric("Series analysed", len(table))
            t2.metric("Avg sleep / series", f"{table['total_sleep_h'].mean():.1f}h")
            t3.metric("Avg efficiency", f"{table['efficiency'].mean():.1f}%")
            # Static st.table avoids the Streamlit 1.28 grid console error.
            st.table(table.set_index("series_id"))
            st.download_button(
                "⬇️ Download batch summary (CSV)",
                table.to_csv(index=False).encode("utf-8"),
                file_name="batch_summary.csv",
                mime="text/csv",
            )

# ---- Compare tab ----
with tab_compare:
    if active_df is None:
        st.info(
            "🆚 **Compare** two series side by side — their sleep totals, "
            "efficiency and timelines.\n\n"
            "👈 Load a file with **2+ series** (e.g. generate synthetic data)."
        )
    else:
        ids = get_series_ids(active_df)
        if len(ids) < 2:
            st.info(
                "This data has only **one** series, so there's nothing to compare. "
                "Use a multi-series file — e.g. **Generate synthetic data** with 2+ "
                "series in the sidebar."
            )
        else:
            cc1, cc2 = st.columns(2)
            a = cc1.selectbox("Series A", ids, index=0, key="cmp_a")
            b = cc2.selectbox("Series B", ids, index=1, key="cmp_b")
            if ctx.get("error") == "model_version":
                _model_version_error()
            elif "submission" not in ctx:
                st.warning("No sleep periods could be derived with the current settings.")
            else:
                prepared, submission = ctx["prepared"], ctx["submission"]
                for slot, (col, s) in enumerate([(cc1, a), (cc2, b)]):
                    with col:
                        m = compute_sleep_metrics(active_df, submission, s)
                        st.metric(f"`{s}` · total sleep", m["total_sleep"])
                        st.metric("Periods / efficiency", f"{m['periods']} · {m['efficiency']}%")
                        # Key includes the slot index so comparing a series with
                        # itself (A == B) still yields unique chart IDs.
                        plot(
                            plotly_prediction(prepared, submission, s),
                            key=f"compare_{slot}_{s}",
                        )

# ---- Model tab ----
with tab_model:
    st.markdown("#### Model card")
    try:
        _, mp = load_pipeline()
        st.caption(f"Serving **`{os.path.basename(mp)}`** — a scikit-learn `Pipeline` "
                   "(ColumnTransformer → XGBoost) predicting `P(awake)`.")
    except Exception:
        st.warning("Model not found.")
    if os.path.exists(METRICS_PATH):
        metrics = json.loads(open(METRICS_PATH).read())
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("CV Avg Precision", f"{metrics['cv_average_precision_mean']:.3f}")
        mc2.metric("OOF ROC-AUC", f"{metrics['oof_roc_auc']:.3f}")
        mc3.metric("OOF Accuracy", f"{metrics['oof_accuracy_at_0.5']:.3f}")
        mc4.metric("Training rows", f"{metrics['n_rows']:,}")
        st.caption(
            f"Trained via **{metrics['trained_via']}** with "
            f"{metrics['cv_folds']}-fold GroupKFold over {metrics['n_series']} series."
        )
        if "feature_importances" in metrics:
            plot(
                plotly_feature_importance(metrics["feature_importances"]),
                key="feature_importance",
            )
    else:
        st.info("No metrics file. Run `python scripts/train_model.py` to generate one.")

# --- About / How it works (always reachable) ---------------------------------
st.divider()
with st.expander("ℹ️ About & how it works"):
    st.markdown(
        """
        The device records two signals every **5 seconds** — **enmo** (movement
        magnitude) and **anglez** (arm angle). The app engineers features from
        these, predicts `P(awake)` with a trained XGBoost pipeline, smooths the
        result and derives **onset** / **wakeup** events. Tune the **detection
        settings** in the sidebar to change sensitivity.
        """
    )
    st.markdown("**Worked example**")
    ex1, ex2, ex3 = st.columns(3)
    ex1.image(Image.open("input_enmo_anglez.png"), caption="Raw features")
    ex2.image(Image.open("input_enmo_target.png"), caption="Labelled sleep (truth)")
    ex3.image(Image.open("input_enmo_target_prediction.png"), caption="Model prediction")

with st.expander("📚 References & acknowledgements"):
    st.write(
        ":gray[CRAN] [Accelerometer data processing with GGIR]"
        "(https://cran.r-project.org/web/packages/GGIR/vignettes/GGIR.html)"
    )
    st.write(
        ":gray[NLM] [Segmenting accelerometer data with unsupervised ML]"
        "(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6326431/)"
    )
    st.write(
        ":gray[Nature Scientific Reports] [Estimating sleep parameters without a sleep diary]"
        "(https://www.nature.com/articles/s41598-018-31266-z)"
    )
    st.caption("Thanks to the Child Mind Institute, Kaggle and Stack Overflow.")

# --- Branding footer ---------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; color:#6b7280; font-size:0.85rem;
                padding:14px 0 4px 0;">
      Built by <b>Akshata Biradar</b> ·
      <a href="https://github.com/akshatabiradars" style="color:#8B5CF6;">GitHub</a> ·
      <a href="https://www.linkedin.com/in/akshata-biradar-bb6306257" style="color:#8B5CF6;">LinkedIn</a>
      &nbsp;·&nbsp; Child Mind Institute — Detect Sleep States
    </div>
    """,
    unsafe_allow_html=True,
)
