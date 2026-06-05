"""Feature engineering, prediction, post-processing and plotting helpers.

These functions back the Streamlit app (``app.py``). Given raw accelerometer
data with the columns ``series_id, step, timestamp, anglez, enmo`` they:

1. validate the data (:func:`check_df`),
2. engineer model features (:func:`feature_engineering`),
3. produce sleep/wake probabilities with the trained pipeline
   (:func:`get_predictions_probas`),
4. post-process the probabilities into discrete ``onset`` / ``wakeup`` events
   and a Kaggle-style submission frame (:func:`build_submission`),
5. summarise the result into clinical metrics and a per-night breakdown
   (:func:`compute_sleep_metrics`, :func:`per_night_breakdown`,
   :func:`batch_summary`), and
6. visualise everything as interactive Plotly charts
   (:func:`plotly_prediction`, :func:`plotly_nightly_trend`,
   :func:`plotly_feature_importance`). The matplotlib helpers
   (:func:`plot_enmo`, :func:`plot_prediction`) are kept for notebook/back-compat
   use.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from pandas.api.types import is_numeric_dtype, is_string_dtype

# --- Tunable post-processing constants ---------------------------------------
# A record is emitted every 5 seconds, so 12 steps == 1 minute.
SECONDS_PER_STEP = 5
STEPS_PER_MINUTE = 12
# Rolling window (in steps) used to smooth raw predictions before deriving
# events. 1000 steps ~= 83 minutes.
SMOOTH_WINDOW = 1000
# Minimum length (in steps) of a sleep period to keep. 12 * 60 == 1 hour.
MIN_SLEEP_STEPS = STEPS_PER_MINUTE * 60
# Lag / rolling window (in steps) used by the difference and rolling features.
FEATURE_PERIODS = 12
# Cap the number of points drawn in interactive charts. A full series can be
# ~100k points; drawing every point makes the first render sluggish, so we plot
# at most this many (evenly subsampled). The model still runs on every row.
MAX_PLOT_POINTS = 6000

# Columns fed to the trained pipeline, in order.
MODEL_FEATURES = [
    "anglez",
    "enmo",
    "enmo_centered",
    "hour",
    "moment",
    "anglez_abs",
    "anglez_diff",
    "enmo_diff",
    "anglez_rolling_mean",
    "enmo_rolling_mean",
    "enmo_x_anglez",
    "enmo_x_anglez_abs",
    "is_weekend",
]

REQUIRED_COLUMNS = ["series_id", "step", "timestamp", "anglez", "enmo"]

# --- Chart palette (StockPeers slate / periwinkle) ---------------------------
# Shared with the app's CSS + the native theme so charts and UI use one palette.
# Keep these in sync with .streamlit/config.toml and the tokens in ``app.py``.
ACCENT = "#615fff"   # periwinkle (sleep spans, bars)
SIGNAL = "#38bdf8"   # sky blue (enmo trace, low end of the actogram)
ONSET = "#01b574"    # success green
WAKEUP = "#fb7185"   # rose
ANGLEZ = "#ffb547"   # amber
CHART_FONT = "Space Grotesk, system-ui, -apple-system, Segoe UI, sans-serif"

# Plotly template used by every chart. Switch with :func:`set_chart_theme`.
CHART_TEMPLATE = "plotly_dark"
# Tracks the active mode so :func:`_base_layout` can pick theme-aware grid lines.
_CHART_DARK = True


def set_chart_theme(dark: bool = True) -> None:
    """Set the Plotly template used by all chart helpers (dark or light)."""
    global CHART_TEMPLATE, _CHART_DARK
    CHART_TEMPLATE = "plotly_dark" if dark else "plotly_white"
    _CHART_DARK = dark


def _base_layout() -> dict:
    """Shared Plotly layout: template, font, transparent bg, themed grid.

    Returned as kwargs for ``fig.update_layout(**_base_layout())``. Apply this
    first; per-chart calls (height, margins, axis titles) run afterwards and
    correctly override these defaults. Grid lines use the slate border colour
    (#314158) to match the native theme.
    """
    grid = "rgba(49,65,88,0.55)" if _CHART_DARK else "rgba(20,24,60,0.10)"
    zero = "rgba(49,65,88,0.85)" if _CHART_DARK else "rgba(20,24,60,0.20)"
    return dict(
        template=CHART_TEMPLATE,
        font=dict(family=CHART_FONT),
        colorway=[ACCENT, SIGNAL, ONSET, WAKEUP, ANGLEZ],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=grid, zerolinecolor=zero),
        yaxis=dict(gridcolor=grid, zerolinecolor=zero),
    )


def line_break(n: int = 3) -> None:
    """Print ``n`` blank lines in the Streamlit app."""
    for _ in range(n):
        st.write("\n")


def check_df(df: pd.DataFrame) -> bool | list[str]:
    """Validate that ``df`` has the required columns with the correct types.

    Expected schema:

    ===========  ===================
    Column       Type
    ===========  ===================
    series_id    str
    step         numeric (int/float)
    timestamp    str
    anglez       numeric (float)
    enmo         numeric (float)
    ===========  ===================

    Returns ``True`` when the data is valid, otherwise a list of error messages.
    """
    if df.shape[0] == 0:
        return ["Dataframe is empty"]

    list_errors: list[str] = []
    columns = df.columns
    numeric_columns = {"step", "anglez", "enmo"}

    for col in REQUIRED_COLUMNS:
        if col not in columns:
            list_errors.append(f"column {col} not found")
            continue
        if col == "series_id" and not is_string_dtype(df[col]):
            list_errors.append(
                f"column {col} has type {df[col].dtype}, expected str"
            )
        if col in numeric_columns and not is_numeric_dtype(df[col]):
            list_errors.append(
                f"column {col} has type {df[col].dtype}, expected int or float"
            )

    return list_errors if list_errors else True


def get_series(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    """Return the portion of ``df`` belonging to ``series_id``."""
    return df[df["series_id"] == series_id]


def get_series_ids(df: pd.DataFrame) -> list[str]:
    """Return the unique ``series_id`` values in ``df``."""
    return df["series_id"].unique().tolist()


def get_moment(hour: int) -> str:
    """Return the moment of day for ``hour``.

    night (0-6), morning (6-12), afternoon (12-18), evening (otherwise).
    """
    if 0 < hour < 6:
        return "night"
    if 6 < hour < 12:
        return "morning"
    if 12 < hour < 18:
        return "afternoon"
    return "evening"


def preprocess_col(
    df: pd.DataFrame, col_name: str, rolling_val: int = 1000
) -> pd.DataFrame:
    """Add a centered, mean-clipped rolling feature for ``col_name``.

    For every ``series_id`` independently this computes (and then keeps only the
    final ``<col>_centered`` column):

    * ``rolling_mean`` - centered rolling mean over ``rolling_val`` steps,
    * ``high_to_mean`` - the rolling mean clipped at its own mean (values above
      the mean are pulled down to it),
    * ``centered`` - ``high_to_mean`` shifted by ``|mean + std| / 2``.

    Returns ``df`` plus the new ``<col>_centered`` column.
    """
    grouped = df.groupby("series_id", sort=False)[col_name]

    rolling_mean = grouped.transform(
        lambda s: s.rolling(window=rolling_val, center=True).mean().bfill().ffill()
    )

    # Clip each series' rolling mean at the series mean.
    series_mean = rolling_mean.groupby(df["series_id"], sort=False).transform("mean")
    high_to_mean = rolling_mean.where(rolling_mean <= series_mean, series_mean)

    # Center each series by |mean + std| / 2.
    htm_by_series = high_to_mean.groupby(df["series_id"], sort=False)
    center_value = (
        htm_by_series.transform("mean") + htm_by_series.transform("std")
    ).abs() / 2

    df_result = df.copy()
    df_result[f"{col_name}_centered"] = high_to_mean - center_value
    return df_result


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time, difference, rolling and interaction features to ``df``.

    Derived columns: ``hour``, ``moment``, ``anglez_abs``, ``anglez_diff``,
    ``enmo_diff``, ``anglez_rolling_mean``, ``enmo_rolling_mean``,
    ``enmo_x_anglez``, ``enmo_x_anglez_abs`` and ``is_weekend`` (plus helper
    columns ``datetime``, ``month`` and ``weekday`` that the caller drops).
    """
    df_result = df.copy()

    # Time features. Parse the *local* wall-clock time: drop the trailing UTC
    # offset before parsing so the result is always a naive datetime64 column.
    # This matters because a series can span a daylight-saving change and thus
    # contain mixed offsets, which would otherwise yield an object-dtype column
    # (breaking the ``.dt`` accessor) and would shift hour/month to UTC.
    local = df["timestamp"].astype(str).str.replace(r"[+-]\d{4}$", "", regex=True)
    df_result["datetime"] = pd.to_datetime(local, utc=False)
    df_result["month"] = df_result["datetime"].dt.month
    df_result["hour"] = df_result["datetime"].dt.hour
    df_result["moment"] = df_result["hour"].apply(get_moment)

    # Absolute anglez.
    df_result["anglez_abs"] = df["anglez"].abs()

    # Lagged differences over FEATURE_PERIODS steps (bfill leading NaNs).
    df_result["anglez_diff"] = (
        df_result.groupby("series_id")["anglez_abs"]
        .diff(periods=FEATURE_PERIODS)
        .bfill()
    )
    df_result["enmo_diff"] = (
        df_result.groupby("series_id")["enmo"].diff(periods=FEATURE_PERIODS).bfill()
    )

    # Centered rolling means (bfill/ffill the centered-window NaNs).
    df_result["anglez_rolling_mean"] = (
        df_result["anglez_abs"]
        .rolling(FEATURE_PERIODS, center=True)
        .mean()
        .bfill()
        .ffill()
    )
    df_result["enmo_rolling_mean"] = (
        df["enmo"].rolling(FEATURE_PERIODS, center=True).mean().bfill().ffill()
    )

    # Interaction terms (vectorised).
    df_result["enmo_x_anglez"] = df_result["enmo"] * df_result["anglez"]
    df_result["enmo_x_anglez_abs"] = df_result["enmo"] * df_result["anglez_abs"]

    # Weekend flag (Monday == 0 ... Sunday == 6).
    df_result["weekday"] = df_result["datetime"].dt.weekday
    df_result["is_weekend"] = (df_result["weekday"] >= 5).astype(int)

    return df_result


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all model features from raw ``enmo``, ``anglez`` and timestamp."""
    nb_series = df["series_id"].nunique()

    # Rolling window: roughly a tenth of the average series length.
    rolling_val = int(df.shape[0] / nb_series / 10)

    df_result = preprocess_col(df, "enmo", rolling_val=rolling_val)
    df_result = get_features(df_result)
    df_result = df_result.drop(columns=["datetime", "month", "weekday"])
    return df_result


def get_predictions_probas(
    df: pd.DataFrame, pipeline, threshold: float = 0.5
) -> tuple[np.ndarray, list[float]]:
    """Return class predictions and per-row confidence from ``pipeline``.

    Predictions are derived from the positive-class probability at ``threshold``
    (default 0.5), so callers can tune the sleep/wake decision boundary. Only the
    model's expected feature columns are passed to the pipeline.
    """
    X = df[MODEL_FEATURES]
    probas = pipeline.predict_proba(X)
    # Positive class = "awake" (class 1); fall back gracefully if single-column.
    pos = probas[:, 1] if probas.ndim == 2 and probas.shape[1] > 1 else probas.ravel()
    y_pred = (pos >= threshold).astype(int)
    y_probas = [max(proba) for proba in probas]
    return y_pred, y_probas


def smooth_results(
    df: pd.DataFrame, y_pred: np.ndarray, smooth_val: int
) -> np.ndarray:
    """Smooth ``y_pred`` with a per-series centered rolling mean, then re-binarise.

    ``df`` provides the ``series_id`` grouping (its index must align with
    ``y_pred``). Returns an integer array of 0/1 predictions.
    """
    preds = pd.Series(np.asarray(y_pred), index=df.index, dtype="float64")
    smoothed = preds.groupby(df["series_id"], sort=False).transform(
        lambda s: s.rolling(window=smooth_val, center=True).mean().bfill().ffill()
    )
    return (smoothed >= 0.5).astype(int).to_numpy()


def get_events(
    df: pd.DataFrame, y_pred: np.ndarray, y_probas: list[float]
) -> pd.DataFrame:
    """Attach ``pred``, ``score`` and ``event`` columns to ``df``.

    ``event`` is ``"onset"`` when the smoothed prediction transitions 1 -> 0,
    ``"wakeup"`` on a 0 -> 1 transition, and ``NaN`` otherwise.
    """
    df_result = df.copy()
    df_result["pred"] = np.asarray(y_pred)
    df_result["score"] = np.asarray(y_probas)

    # diff < 0 (1 -> 0) is an onset; diff > 0 (0 -> 1) is a wakeup; else NaN.
    pred_diff = df_result.groupby("series_id")["pred"].diff().bfill()
    event = np.full(len(df_result), np.nan, dtype=object)
    event[(pred_diff < 0).to_numpy()] = "onset"
    event[(pred_diff > 0).to_numpy()] = "wakeup"
    df_result["event"] = event
    return df_result


def get_submission(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce ``df`` to the Kaggle submission columns with a ``row_id`` index."""
    df = df.dropna()
    df = df[["series_id", "step", "event", "score"]].reset_index(drop=True)
    df.insert(0, "row_id", df.index.values)
    return df


def keep_periods(df: pd.DataFrame, min_period: int) -> pd.DataFrame:
    """Drop sleep and activity periods shorter than ``min_period`` steps.

    Ensures each series starts with an ``onset`` and ends with a ``wakeup``,
    then keeps only onset/wakeup couples whose duration is at least
    ``min_period``.

    Raises ``ValueError`` / ``IndexError`` for a series that lacks a complete
    onset -> wakeup cycle long enough to keep; callers should handle this.
    """
    df_parts = []
    for series_id in df["series_id"].unique():
        df_tmp = get_series(df, series_id)

        pred_onsets = df_tmp[df_tmp["event"] == "onset"]["step"].to_list()
        pred_wakeups = df_tmp[df_tmp["event"] == "wakeup"]["step"].to_list()

        # Drop a leading wakeup / trailing onset so periods are well-formed.
        if min(pred_wakeups) < min(pred_onsets):
            pred_wakeups = pred_wakeups[1:]
        if max(pred_onsets) > max(pred_wakeups):
            pred_onsets = pred_onsets[:-1]

        # Keep sleep periods (onset -> wakeup) at least min_period long.
        kept_onsets = []
        kept_wakeups = []
        for onset, wakeup in zip(pred_onsets, pred_wakeups, strict=False):
            if wakeup - onset >= min_period:
                kept_onsets.append(onset)
                kept_wakeups.append(wakeup)

        # Keep activity periods (wakeup -> next onset) at least min_period long.
        steps_to_keep = [kept_onsets[0]]
        for i, wakeup in enumerate(kept_wakeups[:-1]):
            if kept_onsets[i + 1] - wakeup >= min_period:
                steps_to_keep.append(wakeup)
                steps_to_keep.append(kept_onsets[i + 1])
        steps_to_keep.append(kept_wakeups[-1])

        df_parts.append(df_tmp[df_tmp["step"].isin(steps_to_keep)])

    df_result = pd.concat(df_parts).reset_index(drop=True)
    df_result["row_id"] = df_result.index.values
    return df_result


def build_submission(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_probas: list[float],
    smooth_window: int = SMOOTH_WINDOW,
    min_sleep_steps: int = MIN_SLEEP_STEPS,
) -> pd.DataFrame:
    """Build the Kaggle submission frame from raw predictions and probabilities.

    ``smooth_window`` and ``min_sleep_steps`` are exposed so the app can let
    users tune the post-processing (defaults reproduce the original behaviour).
    """
    y_smooth = smooth_results(df, y_pred, smooth_window)
    df_events = get_events(df, y_smooth, y_probas)
    df_submission = get_submission(df_events)
    df_submission = keep_periods(df_submission, min_sleep_steps)
    return df_submission


def get_random_id(data: pd.DataFrame) -> str:
    """Return a randomly chosen ``series_id`` from ``data``."""
    return np.random.choice(data["series_id"].unique(), size=1)[0]


def plot_enmo(data: pd.DataFrame, series_id: str) -> Figure:
    """Plot ``enmo`` against ``step`` for one series."""
    data_series = get_series(data, series_id)

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(data_series["step"], data_series["enmo"], linewidth=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("enmo")
    fig.suptitle("Evolution of enmo")
    return fig


def plot_prediction(
    df: pd.DataFrame, df_submission: pd.DataFrame, series_id: str
) -> Figure:
    """Plot ``enmo`` with predicted onset (magenta) and wakeup (red) markers."""
    df_viz = get_series(df, series_id)
    df_sub_viz = get_series(df_submission, series_id)
    onset_viz = df_sub_viz[df_sub_viz["event"] == "onset"]
    wakeup_viz = df_sub_viz[df_sub_viz["event"] == "wakeup"]

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(df_viz["step"], df_viz["enmo"], linewidth=0.5)
    ax.vlines(
        x=onset_viz["step"].values,
        ymin=0,
        ymax=4,
        colors="m",
        label="Predicted onset",
        linestyles="dashed",
    )
    ax.vlines(
        x=wakeup_viz["step"].values,
        ymin=0,
        ymax=4,
        colors="r",
        label="Predicted wakeup",
        linestyles="dashed",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("enmo")
    fig.suptitle("Predictions")
    ax.legend()
    return fig


def _sleep_spans(df_submission: pd.DataFrame, series_id: str) -> list[tuple[int, int]]:
    """Return ``(onset_step, wakeup_step)`` pairs for one series, in order."""
    df_sub = get_series(df_submission, series_id).sort_values("step")
    onsets = df_sub[df_sub["event"] == "onset"]["step"].to_list()
    wakeups = df_sub[df_sub["event"] == "wakeup"]["step"].to_list()
    return list(zip(onsets, wakeups, strict=False))


def compute_sleep_summary(df_submission: pd.DataFrame, series_id: str) -> dict:
    """Summarise the predicted sleep for one series into headline KPIs.

    Returns a dict with ``periods`` (number of sleep periods), ``onsets``,
    ``wakeups``, ``total_sleep`` (a human string like ``6h 12m``),
    ``total_sleep_minutes`` and ``avg_confidence`` (0-1).
    """
    spans = _sleep_spans(df_submission, series_id)
    df_sub = get_series(df_submission, series_id)

    # Each step is 5 seconds; (wakeup - onset) steps -> minutes.
    total_minutes = sum((w - o) for o, w in spans) * SECONDS_PER_STEP / 60
    hours, minutes = divmod(int(round(total_minutes)), 60)

    onsets = int((df_sub["event"] == "onset").sum())
    wakeups = int((df_sub["event"] == "wakeup").sum())
    avg_conf = float(df_sub["score"].mean()) if len(df_sub) else 0.0

    return {
        "periods": len(spans),
        "onsets": onsets,
        "wakeups": wakeups,
        "total_sleep": f"{hours}h {minutes:02d}m",
        "total_sleep_minutes": round(total_minutes, 1),
        "avg_confidence": avg_conf,
    }


def _subsample(df: pd.DataFrame, max_points: int = MAX_PLOT_POINTS) -> pd.DataFrame:
    """Evenly subsample ``df`` to at most ``max_points`` rows for fast plotting."""
    if len(df) <= max_points:
        return df
    stride = int(np.ceil(len(df) / max_points))
    return df.iloc[::stride]


def plotly_prediction(
    df: pd.DataFrame,
    df_submission: pd.DataFrame,
    series_id: str,
    show_anglez: bool = False,
    max_points: int = MAX_PLOT_POINTS,
):
    """Interactive timeline: enmo, shaded sleep spans, event markers, and a
    hypnogram band showing the asleep/awake state over time.

    The signal traces are evenly subsampled to ``max_points`` for a fast first
    render (the model still runs on every row). ``show_anglez`` adds a secondary
    anglez trace. Returns a Plotly ``Figure`` with two stacked rows.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    full = get_series(df, series_id).sort_values("step")
    df_viz = _subsample(full, max_points)
    has_time = "timestamp" in df_viz.columns

    def to_x(frame):
        col = pd.to_datetime(frame["timestamp"]) if has_time else frame["step"]
        return col.to_numpy()

    x = to_x(df_viz)
    step_to_x = dict(zip(full["step"].to_numpy(), to_x(full), strict=False))
    y_max = float(full["enmo"].max()) if len(full) else 1.0

    # Two rows: the signal (tall) and a thin hypnogram band sharing the x-axis.
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.82, 0.18],
    )

    fig.add_trace(
        go.Scattergl(
            x=x, y=df_viz["enmo"], mode="lines", name="enmo",
            line=dict(color=SIGNAL, width=1),
            hovertemplate="enmo=%{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )
    if show_anglez and "anglez" in df_viz.columns:
        fig.add_trace(
            go.Scattergl(
                x=x, y=df_viz["anglez"], mode="lines", name="anglez",
                line=dict(color=ANGLEZ, width=0.8), opacity=0.6, yaxis="y3",
                hovertemplate="anglez=%{y:.1f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Shaded predicted-sleep spans (label only the first to avoid Plotly's
    # placeholder "new text" for the rest).
    spans = _sleep_spans(df_submission, series_id)
    for i, (onset_step, wakeup_step) in enumerate(spans):
        ann = (
            dict(annotation_text="sleep", annotation_position="top left")
            if i == 0 else {}
        )
        fig.add_vrect(
            x0=step_to_x.get(onset_step, onset_step),
            x1=step_to_x.get(wakeup_step, wakeup_step),
            fillcolor=ACCENT, opacity=0.16, line_width=0, layer="below",
            row=1, col=1, **ann,
        )

    df_sub = get_series(df_submission, series_id)
    for event_name, color, sym in (
        ("onset", ONSET, "triangle-up"),
        ("wakeup", WAKEUP, "triangle-down"),
    ):
        ev = df_sub[df_sub["event"] == event_name]
        if ev.empty:
            continue
        ev_x = [step_to_x.get(s, s) for s in ev["step"].to_numpy()]
        fig.add_trace(
            go.Scatter(
                x=ev_x, y=[y_max] * len(ev_x), mode="markers",
                name=f"predicted {event_name}",
                marker=dict(color=color, size=10, symbol=sym,
                            line=dict(color="white", width=1)),
                customdata=ev["score"],
                hovertemplate=f"{event_name}<br>conf=%{{customdata:.2f}}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Hypnogram band: 1 while asleep (inside a span), 0 otherwise, over the
    # subsampled steps.
    steps = df_viz["step"].to_numpy()
    asleep = np.zeros(len(steps), dtype=int)
    for onset_step, wakeup_step in spans:
        asleep |= ((steps >= onset_step) & (steps <= wakeup_step)).astype(int)
    fig.add_trace(
        go.Scatter(
            x=x, y=asleep, mode="lines", name="state", line_shape="hv",
            line=dict(color=ACCENT, width=1), fill="tozeroy",
            fillcolor="rgba(97,95,255,0.35)", showlegend=False,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=np.where(asleep == 1, "asleep", "awake"),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        **_base_layout(),
        title=f"enmo with predicted sleep — {series_id}",
        height=470,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="enmo", row=1, col=1)
    fig.update_yaxes(
        title_text="state", row=2, col=1,
        tickmode="array", tickvals=[0, 1], ticktext=["awake", "asleep"],
        range=[-0.1, 1.1],
    )
    fig.update_xaxes(title_text="time" if has_time else "step", row=2, col=1)
    return fig


def _step_to_local_time(df: pd.DataFrame, series_id: str) -> dict:
    """Map each ``step`` to its local wall-clock ``Timestamp`` for one series."""
    s = get_series(df, series_id)
    if "timestamp" not in s.columns:
        return {}
    local = s["timestamp"].astype(str).str.replace(r"[+-]\d{4}$", "", regex=True)
    times = pd.to_datetime(local, utc=False)
    return dict(zip(s["step"].to_numpy(), times.to_numpy(), strict=False))


def per_night_breakdown(
    df: pd.DataFrame, df_submission: pd.DataFrame, series_id: str
) -> pd.DataFrame:
    """One row per detected sleep period with clock times and duration.

    Columns: ``night``, ``onset``, ``wakeup``, ``duration_h``, ``confidence``.
    """
    spans = _sleep_spans(df_submission, series_id)
    step_time = _step_to_local_time(df, series_id)
    df_sub = get_series(df_submission, series_id)
    score_by_step = dict(zip(df_sub["step"].to_numpy(), df_sub["score"].to_numpy(), strict=False))

    rows = []
    for i, (onset_step, wakeup_step) in enumerate(spans, start=1):
        onset_t = step_time.get(onset_step)
        wakeup_t = step_time.get(wakeup_step)
        dur_h = (wakeup_step - onset_step) * SECONDS_PER_STEP / 3600
        confs = [score_by_step.get(onset_step), score_by_step.get(wakeup_step)]
        confs = [c for c in confs if c is not None]
        rows.append(
            {
                "night": i,
                "onset": pd.Timestamp(onset_t) if onset_t is not None else onset_step,
                "wakeup": pd.Timestamp(wakeup_t) if wakeup_t is not None else wakeup_step,
                "duration_h": round(dur_h, 2),
                "confidence": round(float(np.mean(confs)), 3) if confs else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_sleep_metrics(
    df: pd.DataFrame, df_submission: pd.DataFrame, series_id: str
) -> dict:
    """Clinical-style sleep metrics for one series.

    Adds to :func:`compute_sleep_summary`:

    * ``efficiency`` - % of the recording's span spent asleep,
    * ``latency_h`` - hours from recording start to the first onset,
    * ``avg_period_h`` / ``longest_h`` / ``shortest_h`` - per-period durations,
    * ``awakenings`` - number of wake transitions between sleep periods.
    """
    base = compute_sleep_summary(df_submission, series_id)
    spans = _sleep_spans(df_submission, series_id)
    s = get_series(df, series_id)

    durations_h = [(w - o) * SECONDS_PER_STEP / 3600 for o, w in spans]
    total_h = sum(durations_h)

    # Recording span in hours (step range * 5s).
    if len(s):
        span_steps = s["step"].max() - s["step"].min()
        recording_h = span_steps * SECONDS_PER_STEP / 3600
    else:
        recording_h = 0.0

    # Latency: start of recording -> first onset.
    latency_h = (
        (spans[0][0] - s["step"].min()) * SECONDS_PER_STEP / 3600 if spans else np.nan
    )

    base.update(
        {
            "recording_h": round(recording_h, 2),
            "efficiency": round(100 * total_h / recording_h, 1) if recording_h else 0.0,
            "latency_h": round(float(latency_h), 2) if spans else None,
            "avg_period_h": round(float(np.mean(durations_h)), 2) if durations_h else 0.0,
            "longest_h": round(float(np.max(durations_h)), 2) if durations_h else 0.0,
            "shortest_h": round(float(np.min(durations_h)), 2) if durations_h else 0.0,
            "awakenings": max(0, len(spans) - 1),
        }
    )
    return base


def recording_span(df: pd.DataFrame, series_id: str) -> dict:
    """Return the date range and length of one series' recording.

    Keys: ``start`` / ``end`` (``Timestamp`` or ``None``), ``days`` (int) and
    ``label`` (a short human string like ``Aug 14 → Aug 20, 2018 · 6 days``).
    Falls back to step counts when no usable ``timestamp`` is present.
    """
    s = get_series(df, series_id).sort_values("step")
    n_steps = len(s)
    approx_days = max(1, round(n_steps * SECONDS_PER_STEP / 86400))

    if "timestamp" not in s.columns or s.empty:
        return {"start": None, "end": None, "days": approx_days,
                "label": f"~{approx_days} day(s)"}

    local = s["timestamp"].astype(str).str.replace(r"[+-]\d{4}$", "", regex=True)
    times = pd.to_datetime(local, utc=False)
    start, end = times.iloc[0], times.iloc[-1]
    days = max(1, (end.normalize() - start.normalize()).days + 1)
    if start.year == end.year:
        label = f"{start:%b %d} → {end:%b %d}, {end.year} · {days} day(s)"
    else:
        label = f"{start:%b %d %Y} → {end:%b %d %Y} · {days} day(s)"
    return {"start": start, "end": end, "days": days, "label": label}


def rate_metric(kind: str, value) -> tuple[str, str]:
    """Map a sleep metric to a (label, streamlit-delta-color) hint.

    ``kind`` is one of ``"efficiency"`` (%), ``"latency"`` (hours) or
    ``"awakenings"`` (count). Returns e.g. ``("good", "normal")`` where the
    second item is suitable for ``st.metric(delta_color=...)`` (``"normal"`` is
    green, ``"inverse"`` red, ``"off"`` grey).
    """
    if value is None:
        return ("—", "off")
    if kind == "efficiency":
        if value >= 85:
            return ("good", "normal")
        if value >= 75:
            return ("fair", "off")
        return ("low", "inverse")
    if kind == "latency":
        if value <= 0.5:
            return ("good", "normal")
        if value <= 1.0:
            return ("fair", "off")
        return ("long", "inverse")
    if kind == "awakenings":
        if value <= 1:
            return ("good", "normal")
        if value <= 3:
            return ("fair", "off")
        return ("high", "inverse")
    return ("", "off")


def batch_summary(
    df: pd.DataFrame, df_submission: pd.DataFrame
) -> pd.DataFrame:
    """One summary row per series in ``df`` (for the batch dashboard).

    Series for which no sleep period could be derived are skipped gracefully.
    """
    rows = []
    sub_ids = set(df_submission["series_id"].unique())
    for sid in get_series_ids(df):
        if sid not in sub_ids:
            rows.append({"series_id": sid, "periods": 0, "total_sleep_h": 0.0,
                         "efficiency": 0.0, "avg_confidence": np.nan})
            continue
        m = compute_sleep_metrics(df, df_submission, sid)
        rows.append(
            {
                "series_id": sid,
                "periods": m["periods"],
                "total_sleep_h": round(m["total_sleep_minutes"] / 60, 2),
                "efficiency": m["efficiency"],
                "avg_confidence": round(m["avg_confidence"], 3),
            }
        )
    return pd.DataFrame(rows)


def plotly_actogram(df: pd.DataFrame, df_submission: pd.DataFrame, series_id: str):
    """Actogram-style heatmap: rows = days, columns = hour-of-day, shaded by sleep.

    Each cell is the fraction of that (day, hour) the model predicts asleep. Needs
    a ``timestamp`` column; returns ``None`` if unavailable.
    """
    import plotly.graph_objects as go

    s = get_series(df, series_id).sort_values("step")
    if "timestamp" not in s.columns or s.empty:
        return None

    local = s["timestamp"].astype(str).str.replace(r"[+-]\d{4}$", "", regex=True)
    times = pd.to_datetime(local, utc=False)
    steps = s["step"].to_numpy()

    # Mark asleep steps from the predicted spans.
    asleep = np.zeros(len(steps), dtype=int)
    for onset_step, wakeup_step in _sleep_spans(df_submission, series_id):
        asleep |= ((steps >= onset_step) & (steps <= wakeup_step)).astype(int)

    grid = pd.DataFrame(
        {"date": times.dt.date, "hour": times.dt.hour, "asleep": asleep}
    )
    # Fraction asleep per (date, hour).
    pivot = (
        grid.groupby(["date", "hour"])["asleep"].mean().unstack(fill_value=0.0)
        .reindex(columns=range(24), fill_value=0.0)
    )
    if pivot.empty:
        return None

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}" for h in pivot.columns],
            y=[str(d) for d in pivot.index],
            # Sky -> periwinkle ramp: faint at "awake", saturated accent at "asleep".
            colorscale=[[0, "rgba(56,189,248,0.06)"], [0.5, "#38bdf8"], [1, ACCENT]],
            zmin=0, zmax=1,
            colorbar=dict(title="asleep", tickformat=".0%"),
            hovertemplate="%{y} · %{x}:00<br>%{z:.0%} asleep<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(),
        title="Sleep regularity (actogram)",
        xaxis_title="hour of day",
        yaxis_title="date",
        height=max(220, 60 + 26 * len(pivot.index)),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plotly_nightly_trend(per_night: pd.DataFrame):
    """Bar chart of sleep duration per night (for the trends view)."""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Bar(
            x=per_night["night"],
            y=per_night["duration_h"],
            marker_color=ACCENT,
            customdata=per_night["confidence"],
            hovertemplate="night %{x}<br>%{y:.2f} h<br>conf=%{customdata:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(),
        title="Sleep duration per night",
        xaxis_title="night",
        yaxis_title="hours asleep",
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plotly_feature_importance(importances: dict):
    """Horizontal bar chart of feature importances (highest at the top)."""
    import plotly.graph_objects as go

    # Sort ascending so the largest bar sits at the top of a horizontal chart.
    items = sorted(importances.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=ACCENT,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(),
        title="Feature importances",
        xaxis_title="importance",
        height=420,
        margin=dict(l=10, r=20, t=50, b=40),
    )
    return fig


def build_html_report(series_id: str, metrics: dict, per_night: pd.DataFrame) -> str:
    """A self-contained HTML report (open in a browser, Print -> Save as PDF)."""
    rows = "".join(
        f"<tr><td>{r.night}</td><td>{r.onset}</td><td>{r.wakeup}</td>"
        f"<td>{r.duration_h:.2f} h</td><td>{r.confidence:.2f}</td></tr>"
        for r in per_night.itertuples()
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Sleep report — {series_id}</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;max-width:820px;margin:32px auto;color:#1a1a2e}}
 h1{{color:#615fff}} .kpis{{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}}
 .card{{flex:1;min-width:140px;border:1px solid #ddd;border-radius:10px;padding:12px 14px}}
 .card b{{display:block;font-size:1.4rem;color:#615fff}}
 table{{width:100%;border-collapse:collapse;margin-top:12px}}
 th,td{{border-bottom:1px solid #eee;padding:8px;text-align:left;font-size:.92rem}}
 th{{color:#666}} .foot{{margin-top:24px;color:#888;font-size:.8rem}}
</style></head><body>
<h1>😴 Sleep report</h1>
<p>Series <code>{series_id}</code> · Child Mind Institute — Detect Sleep States</p>
<div class="kpis">
 <div class="card"><b>{metrics['periods']}</b>sleep periods</div>
 <div class="card"><b>{metrics['total_sleep']}</b>total sleep</div>
 <div class="card"><b>{metrics['efficiency']}%</b>efficiency</div>
 <div class="card"><b>{metrics['awakenings']}</b>awakenings</div>
 <div class="card"><b>{metrics['avg_confidence'] * 100:.0f}%</b>avg confidence</div>
</div>
<table><thead><tr><th>Night</th><th>Onset</th><th>Wakeup</th><th>Duration</th><th>Confidence</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="foot">Generated by the Detect Sleep States app. Tip: use your browser's
Print dialog and choose "Save as PDF".</p>
</body></html>"""


def generate_synthetic_series(
    n_series: int = 1, days: int = 4, seed: int | None = None
) -> pd.DataFrame:
    """Generate synthetic accelerometer data in the app's schema for testing.

    Produces a believable day (awake / high movement) vs. night (asleep / low
    movement) pattern so the model detects real sleep periods. Each series gets a
    Kaggle-style 12-char hex ``series_id``.
    """
    rng = np.random.default_rng(seed)
    steps_per_series = days * 24 * 60 * 60 // SECONDS_PER_STEP

    frames = []
    for s in range(n_series):
        sid = "".join(rng.choice(list("0123456789abcdef"), size=12))
        start = pd.Timestamp("2024-03-11T14:00:00") + pd.Timedelta(days=s)
        ts = start + pd.to_timedelta(np.arange(steps_per_series) * SECONDS_PER_STEP, unit="s")
        hour = ts.hour + ts.minute / 60.0
        awake = ~((hour >= 23) | (hour < 7))

        enmo = np.where(
            awake,
            np.abs(rng.normal(0.18, 0.12, steps_per_series))
            + rng.gamma(1.5, 0.06, steps_per_series),
            np.abs(rng.normal(0.005, 0.008, steps_per_series)),
        )
        spikes = rng.random(steps_per_series) < 0.01
        enmo[spikes & awake] += rng.uniform(0.5, 2.5, int((spikes & awake).sum()))
        enmo = np.clip(enmo, 0, None).round(4)
        anglez = np.where(
            awake,
            rng.normal(0, 35, steps_per_series),
            rng.normal(-45, 8, steps_per_series),
        ).round(4)

        frames.append(
            pd.DataFrame(
                {
                    "series_id": sid,
                    "step": np.arange(steps_per_series),
                    "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S-0400"),
                    "anglez": anglez,
                    "enmo": enmo,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def read_file(filename: str, file) -> pd.DataFrame | str:
    """Read a ``.csv`` or ``.parquet`` file, or return an error message."""
    _, file_extension = os.path.splitext(filename)
    if file_extension == ".csv":
        return pd.read_csv(file)
    if file_extension == ".parquet":
        return pd.read_parquet(file)
    return "Incorrect file extension"
