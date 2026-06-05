"""Streamlit web app for the Child Mind Institute - Detect Sleep States project.

Loads a trained scikit-learn pipeline and detects sleep ``onset`` / ``wakeup``
events from wrist-worn accelerometer data (``enmo`` and ``anglez``, every 5 s).

The app is a multi-page app (native ``st.navigation``) with a sidebar menu:

* **Dashboard**          - single-series KPIs, interactive chart, per-night
  breakdown and downloads, plus **Batch** and **Compare** tabs.
* **Data**               - choose a sample / upload / generate the dataset.
* **Detection settings** - tune the threshold, smoothing and minimum sleep period.
* **Model**              - model card with validation metrics and importances.
* **About** / **References** - how it works and acknowledgements.

The chosen dataset and the detection settings are shared across pages via
``st.session_state`` (see :func:`_ensure_defaults` and :func:`_detection_params`).

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

# Default detection-slider values (module level so every page + the reset button
# can reach them, and so :func:`_ensure_defaults` can seed session_state).
DEFAULTS = {
    "threshold": 0.50,
    "smooth_min": SMOOTH_WINDOW // STEPS_PER_MINUTE,
    "min_sleep_min": MIN_SLEEP_STEPS // STEPS_PER_MINUTE,
}

st.set_page_config(
    page_title="Detect Sleep States",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme: the app is dark-only. Set the Plotly template to dark to match the
# slate palette configured natively in .streamlit/config.toml.
set_chart_theme(dark=True)

# --- Light-touch styling -----------------------------------------------------
# The bulk of the look (slate palette, Space Grotesk font, widget borders, corner
# radius) is handled by Streamlit's NATIVE theming in .streamlit/config.toml, so
# this CSS only adds the few pieces native theming can't: a flat accented hero,
# an underline on the selected tab, rating pills and section headers. We reuse a
# small set of tokens so the values stay in sync with the config palette.
st.markdown(
    """
    <style>
      :root {
        --accent:#615fff; --bg:#1d293d; --surface:#0f172b;
        --border:#314158; --text:#e2e8f0; --text-muted:#94a3b8;
        --good:#01b574; --warn:#ffb547; --bad:#fb7185;
      }

      .block-container { padding-top: 1.4rem; }

      /* Hero: flat slate panel with an accent rule, no gradients */
      .hero {
        background:var(--surface); border:1px solid var(--border);
        border-left:4px solid var(--accent);
        border-radius:0.6rem; padding:22px 26px; margin-bottom:14px;
      }
      .hero h1 { margin:0; font-size:1.7rem; font-weight:400; color:var(--text); }
      .hero p  { margin:6px 0 0 0; color:var(--text-muted); }

      /* Tabs: clean underline on the active tab (StockPeers-style) */
      div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        color:var(--accent);
      }

      /* Rating pills */
      .pill {
        display:inline-block; border-radius:999px; padding:2px 11px;
        font-size:0.72rem; font-weight:500; letter-spacing:0.2px;
      }
      .pill-good { background:rgba(1,181,116,0.16); color:var(--good); }
      .pill-fair { background:rgba(255,181,71,0.16); color:var(--warn); }
      .pill-low  { background:rgba(251,113,133,0.16); color:var(--bad); }

      /* Section headers */
      .section-head { margin:6px 0 4px 0; }
      .section-head h3 { margin:0; font-size:1.1rem; font-weight:400; color:var(--text); }
      .section-head p  { margin:2px 0 0 0; font-size:0.85rem; color:var(--text-muted); }
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


# `border=` was added to st.container in Streamlit 1.29. On older versions we
# fall back to a plain container (the CSS still styles its inner block), so the
# app keeps working on the pinned-floor Streamlit 1.28.
_CONTAINER_SUPPORTS_BORDER = "border" in inspect.signature(st.container).parameters


def card():
    """A bordered card container where supported, else a plain container."""
    if _CONTAINER_SUPPORTS_BORDER:
        return st.container(border=True)
    return st.container()


def section(title: str, subtitle: str = ""):
    """Render a consistent section header (title + optional subtitle)."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="section-head"><h3>{title}</h3>{sub}</div>',
        unsafe_allow_html=True,
    )


def pill(label: str, kind: str) -> str:
    """Return an HTML rating chip; ``kind`` is one of good/fair/low/normal."""
    css = {"good": "pill-good", "fair": "pill-fair", "low": "pill-low"}.get(kind, "")
    return f'<span class="pill {css}">{label}</span>'


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


def _model_version_error():
    import sklearn

    st.error(
        f"❌ The model could not run with scikit-learn {sklearn.__version__}. "
        "Re-train a portable model with `python scripts/train_model.py`, or run "
        "in the bundled Python 3.11 environment (`.venv311`)."
    )


# --- Shared state ------------------------------------------------------------
# Because only ONE page function runs per rerun, the Data and Detection-settings
# pages persist their choices into st.session_state and the Dashboard reads them.
# We seed every key on every rerun so a value is never lost when its widget is
# not rendered on the current page (and so the Dashboard shows results on first
# load with no interaction).
def _ensure_defaults():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    if "active_df" not in st.session_state:
        st.session_state["active_df"] = load_sample(SAMPLE_SERIES["Sample 1"])
        st.session_state["data_label"] = "Sample 1"


def _detection_params():
    """Read the detection settings safely from any page."""
    threshold = st.session_state.get("threshold", DEFAULTS["threshold"])
    smooth_min = st.session_state.get("smooth_min", DEFAULTS["smooth_min"])
    min_sleep_min = st.session_state.get("min_sleep_min", DEFAULTS["min_sleep_min"])
    return (
        threshold,
        smooth_min * STEPS_PER_MINUTE,
        min_sleep_min * STEPS_PER_MINUTE,
    )


# --- Pages -------------------------------------------------------------------
def dashboard_page():
    df = st.session_state.get("active_df")
    if df is None:
        st.info(
            "No data loaded yet. Open the **Data** page in the sidebar to pick a "
            "sample, upload your own file, or generate synthetic data."
        )
        _data_page = st.session_state.get("_data_page")
        if _data_page is not None and hasattr(st, "page_link"):
            st.page_link(_data_page, label="Go to Data", icon=":material/database:")
        return

    data_label = st.session_state.get("data_label", "")
    threshold, smooth_window, min_sleep_steps = _detection_params()

    # Compute the analysis ONCE (cached) before the tabs, with a visible status so
    # the main area never sits blank while the model runs.
    with st.status(
        f"Analysing {len(df):,} readings · {df['series_id'].nunique()} series…",
        expanded=False,
    ) as status:
        st.write("Engineering features…")
        ctx = analyze(df, threshold, smooth_window, min_sleep_steps)
        if ctx.get("error") == "model_version":
            status.update(label="Model could not load", state="error")
        elif ctx.get("error") == "no_periods":
            status.update(label="No sleep periods found with current settings", state="error")
        else:
            status.update(label="Analysis complete ✓", state="complete")

    tab_analyze, tab_batch, tab_compare = st.tabs(["🔍 Analyze", "📊 Batch", "🆚 Compare"])

    # ---- Analyze tab ----
    with tab_analyze:
        series_ids = get_series_ids(df)
        sid = st.selectbox("Series to view", series_ids, key="analyze_series")
        show_anglez = st.checkbox("Show anglez trace", value=False, key="show_anglez")
        span = recording_span(df, sid)
        st.caption(f"Showing **{data_label}** · series `{sid}` · 📅 {span['label']}")

        if ctx.get("error") == "model_version":
            _model_version_error()
        elif ctx.get("error") == "no_periods":
            st.warning(
                "⚠️ No complete sleep period (onset → wakeup) longer than the minimum "
                "was found for this series. Try lowering the **minimum sleep period** "
                "or the **threshold** on the Detection settings page."
            )
        else:
            prepared, submission = ctx["prepared"], ctx["submission"]
            m = compute_sleep_metrics(df, submission, sid)

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

            per_night = per_night_breakdown(df, submission, sid)
            with card():
                section("Per-night breakdown", "Onset, wakeup and duration for each detected sleep period.")
                left, right = st.columns([3, 2])
                with left:
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
        section("Summary across all series", "One row per series — total sleep, efficiency and confidence.")
        if ctx.get("error") == "model_version":
            _model_version_error()
        elif "submission" not in ctx:
            st.warning("No sleep periods could be derived for any series with the current settings.")
        else:
            table = batch_summary(df, ctx["submission"])
            t1, t2, t3 = st.columns(3)
            t1.metric("Series analysed", len(table))
            t2.metric("Avg sleep / series", f"{table['total_sleep_h'].mean():.1f}h")
            t3.metric("Avg efficiency", f"{table['efficiency'].mean():.1f}%")
            with card():
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
        ids = get_series_ids(df)
        if len(ids) < 2:
            st.info(
                "This data has only **one** series, so there's nothing to compare. "
                "Use a multi-series file — e.g. **Generate synthetic data** with 2+ "
                "series on the Data page."
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
                    with col, card():
                        m = compute_sleep_metrics(df, submission, s)
                        st.metric(f"`{s}` · total sleep", m["total_sleep"])
                        st.metric("Periods / efficiency", f"{m['periods']} · {m['efficiency']}%")
                        # Key includes the slot index so comparing a series with
                        # itself (A == B) still yields unique chart IDs.
                        plot(
                            plotly_prediction(prepared, submission, s),
                            key=f"compare_{slot}_{s}",
                        )


def data_page():
    section("Choose your data", "Pick a bundled sample, upload your own file, or generate synthetic data.")
    source = st.radio(
        "Data source",
        ["Use a sample series", "Upload my own data", "Generate synthetic data"],
        key="data_source",
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
        if st.button("✨ Generate"):
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
            )
    else:
        choice = st.selectbox("Sample series", list(SAMPLE_SERIES.keys()), key="sample_choice")
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

    # Persist the choice for the other pages, and preview it.
    st.session_state["active_df"] = active_df
    st.session_state["data_label"] = data_label

    if active_df is not None:
        with card():
            section(f"Preview · {len(active_df):,} rows", data_label)
            st.dataframe(active_df.head(8), use_container_width=True, hide_index=True)
        st.success("Data loaded. Open the **Dashboard** page to see the analysis.")
    else:
        st.info("No valid data selected yet.")


def settings_page():
    section("Detection settings", "Tune how sensitively the app turns predictions into sleep events.")
    if st.button("↺ Reset to defaults"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    with card():
        # No default value arg: the key is pre-seeded by _ensure_defaults(), so
        # session_state drives the slider (and avoids Streamlit's
        # "widget default value + session_state" warning).
        st.slider(
            "Sleep/wake threshold", 0.05, 0.95, step=0.05,
            key="threshold",
            help="Higher = the model must be more confident the child is awake.",
        )
        st.slider(
            "Smoothing window (minutes)", 5, 180, step=5,
            key="smooth_min",
            help="Rolling window used to remove short flickers before detecting events.",
        )
        st.slider(
            "Minimum sleep period (minutes)", 15, 240, step=15,
            key="min_sleep_min",
            help="Discard any detected sleep period shorter than this.",
        )
    st.caption("Changes apply to the **Dashboard** the next time you open it.")


def model_page():
    with card():
        section("Model card", "Validation metrics and feature importances for the served pipeline.")
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


def about_page():
    section("About & how it works")

    # --- The big picture ----------------------------------------------------
    with card():
        section("In one sentence")
        st.markdown(
            """
            This app looks at the tiny movements of a **wrist-worn watch** and
            works out **when a child fell asleep and when they woke up** — without
            anyone having to keep a sleep diary.
            """
        )
        st.markdown(
            """
            Think of it like this: when you're **awake**, your wrist is constantly
            moving — typing, walking, fidgeting. When you're **asleep**, your wrist
            goes almost completely still for hours. The app learns to spot that
            difference automatically.
            """
        )

    # --- What the watch records --------------------------------------------
    with card():
        section("Step 1 · What the watch measures")
        st.markdown(
            """
            The watch takes **two readings every 5 seconds**, all night and all day:

            - **Movement** *(called `enmo`)* — how much the wrist is moving right
              now. A high number means lots of motion; a number near zero means the
              wrist is still.
            - **Arm angle** *(called `anglez`)* — which way the arm is tilted. Your
              arm rests in different positions when you sleep versus when you're up
              and about.

            Over a few days that's **hundreds of thousands of readings** — far too
            many for a person to scan by eye. That's the job the app takes over.
            """
        )

    # --- How the app decides asleep vs awake -------------------------------
    with card():
        section("Step 2 · How the app decides asleep vs. awake")
        st.markdown(
            """
            1. **It studies the pattern.** For every 5-second moment, the app looks
               not just at that instant but at the **trend around it** — the time of
               day, whether movement has been rising or falling, the average over the
               last minute, and so on.
            2. **It makes a prediction.** A trained model (think of it as a very
               experienced reader of these patterns) gives each moment a score: *how
               likely is it the child is awake right now?*
            3. **It draws the line.** Anything above the **threshold** is called
               "awake", anything below is "asleep". You can move this line on the
               **Detection settings** page to be stricter or more relaxed.
            4. **It tidies up the result.** Real data is noisy — a sleeping child
               might twitch for a few seconds. The app **smooths** the result so
               one brief wiggle doesn't get mistaken for waking up.
            5. **It marks the moments that matter.** When the tidied-up signal flips
               from awake to asleep it records a **sleep onset**; when it flips back
               it records a **wake-up**.
            6. **It ignores the tiny stretches.** Finally it throws away any sleep
               (or awake) stretch shorter than the **minimum period** you set, so
               you're left with real nights of sleep, not noise.
            """
        )

    # --- Worked example -----------------------------------------------------
    with card():
        section("See it in action", "The same recording at three stages.")
        ex1, ex2, ex3 = st.columns(3)
        ex1.image(Image.open("input_enmo_anglez.png"),
                  caption="1. The raw movement & angle signals from the watch")
        ex2.image(Image.open("input_enmo_target.png"),
                  caption="2. The true sleep periods (shaded)")
        ex3.image(Image.open("input_enmo_target_prediction.png"),
                  caption="3. What the app predicts — onset ▲ and wake-up ▼")

    # --- What you get on the Dashboard -------------------------------------
    with card():
        section("Step 3 · What you see on the Dashboard")
        st.markdown(
            """
            Once the app has found the sleep periods, it turns them into easy-to-read
            numbers and charts:

            - **Total sleep** — how long the child actually slept.
            - **Efficiency** — of the time in bed, what share was real sleep
              *(higher is better; **≥85%** is good)*.
            - **Sleep latency** — how long it took to fall asleep after the recording
              started.
            - **Awakenings** — how many times sleep was interrupted.
            - A **timeline chart** showing the movement signal with sleep periods
              shaded, and the per-night breakdown underneath.
            """
        )

    # --- Plain-language glossary -------------------------------------------
    with st.expander("📖 Glossary — the words explained simply"):
        st.markdown(
            """
            | Term | What it means in plain words |
            |---|---|
            | **enmo** | How much the wrist is moving (movement strength). |
            | **anglez** | The tilt/angle of the arm. |
            | **onset** | The moment the child *falls asleep*. |
            | **wake-up** | The moment the child *wakes up*. |
            | **threshold** | The cut-off the app uses to call a moment "awake". |
            | **smoothing** | Averaging out brief blips so they don't fool the app. |
            | **efficiency** | The share of time-in-bed that was actually sleep. |
            | **latency** | How long it took to fall asleep. |
            | **confidence** | How sure the app is about a prediction (0–100%). |
            """
        )

    # --- Quick FAQ ----------------------------------------------------------
    with st.expander("❓ Common questions"):
        st.markdown(
            """
            **Do I need my own data to try it?**
            No — open the **Data** page and pick a built-in *sample*, or *generate
            synthetic data* to explore the app instantly.

            **Is this a medical diagnosis?**
            No. It's a research/educational tool that estimates sleep from movement.
            It is not a substitute for professional sleep assessment.

            **Why did it miss a nap / split a night?**
            The result depends on the **Detection settings**. If short naps are being
            dropped, lower the *minimum sleep period*; if the app is too eager to call
            "awake", lower the *threshold*. Changes take effect on the Dashboard.

            **Where does the data come from?**
            From the Kaggle *Child Mind Institute — Detect Sleep States* competition,
            which uses real wrist-worn accelerometer recordings. See the
            **references below** for the science behind it.
            """
        )

    # --- References & acknowledgements -------------------------------------
    with card():
        section("References & acknowledgements", "The science and sources behind the app.")
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


# --- App shell ---------------------------------------------------------------
_ensure_defaults()

# Hero (rendered once, on every page).
st.markdown(
    """
    <div class="hero">
      <h1>😴 Child Mind Institute — Detect Sleep States</h1>
      <p>Detection of sleep onset and wake from wrist-worn accelerometer data ⌚</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Native multi-page navigation (Material Symbols icons, React-style).
# References & acknowledgements live inside the About page, not as a separate page.
page_dashboard = st.Page(dashboard_page, title="Dashboard", icon=":material/dashboard:", default=True)
page_data = st.Page(data_page, title="Data", icon=":material/database:")
page_settings = st.Page(settings_page, title="Detection settings", icon=":material/tune:")
page_model = st.Page(model_page, title="Model", icon=":material/neurology:")
page_about = st.Page(about_page, title="About", icon=":material/info:")
# Stash the Data page so the Dashboard's empty state can link to it.
st.session_state["_data_page"] = page_data

nav = st.navigation(
    {
        "Analysis": [page_dashboard, page_data, page_settings, page_model],
        "Info": [page_about],
    }
)

# Shared sidebar footer beneath the nav menu (model name + links).
with st.sidebar:
    st.divider()
    _model_path = "—"
    try:
        _, _model_path = load_pipeline()
    except Exception:
        pass
    st.caption(f"Model: `{os.path.basename(_model_path)}`")
    st.caption(f"Data: **{st.session_state.get('data_label', '—') or '—'}**")
    st.caption(
        "[Kaggle competition](https://www.kaggle.com/competitions/"
        "child-mind-institute-detect-sleep-states) · "
        "[Child Mind Institute](https://childmind.org/)"
    )

nav.run()

# --- Branding footer ---------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; color:var(--text-muted); font-size:0.85rem;
                padding:14px 0 4px 0;">
      Built by <b>Akshata Biradar</b> ·
      <a href="https://github.com/akshatabiradars" style="color:var(--accent);">GitHub</a> ·
      <a href="https://www.linkedin.com/in/akshata-biradar-bb6306257" style="color:var(--accent);">LinkedIn</a>
      &nbsp;·&nbsp; Child Mind Institute — Detect Sleep States
    </div>
    """,
    unsafe_allow_html=True,
)
