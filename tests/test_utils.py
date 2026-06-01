"""Unit tests for the pure helper functions in ``utils.py``."""

import numpy as np
import pandas as pd
import pytest

import utils


# --- get_moment --------------------------------------------------------------
@pytest.mark.parametrize(
    "hour,expected",
    [
        (3, "night"),
        (9, "morning"),
        (15, "afternoon"),
        (20, "evening"),
        (0, "evening"),  # boundary: not 0 < hour
        (6, "evening"),  # boundary: not 6 < hour < 12 and not >... -> evening
    ],
)
def test_get_moment(hour, expected):
    assert utils.get_moment(hour) == expected


# --- check_df ----------------------------------------------------------------
def _valid_df():
    return pd.DataFrame(
        {
            "series_id": ["a", "a", "a"],
            "step": [0, 1, 2],
            "timestamp": [
                "2018-08-14T15:30:00-0400",
                "2018-08-14T15:30:05-0400",
                "2018-08-14T15:30:10-0400",
            ],
            "anglez": [2.6, 2.7, 2.8],
            "enmo": [0.02, 0.03, 0.01],
        }
    )


def test_check_df_valid():
    assert utils.check_df(_valid_df()) is True


def test_check_df_empty():
    assert utils.check_df(_valid_df().iloc[0:0]) == ["Dataframe is empty"]


def test_check_df_missing_column():
    df = _valid_df().drop(columns=["enmo"])
    errors = utils.check_df(df)
    assert errors is not True
    assert any("enmo" in e for e in errors)


def test_check_df_wrong_type():
    df = _valid_df()
    df["enmo"] = df["enmo"].astype(str)  # should be numeric
    errors = utils.check_df(df)
    assert errors is not True
    assert any("enmo" in e for e in errors)


# --- read_file ---------------------------------------------------------------
def test_read_file_csv(tmp_path):
    df = _valid_df()
    p = tmp_path / "series.csv"
    df.to_csv(p, index=False)
    with open(p, "rb") as f:
        out = utils.read_file("series.csv", f)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(df.columns)


def test_read_file_bad_extension():
    assert utils.read_file("series.txt", None) == "Incorrect file extension"


# --- get_series / get_series_ids ---------------------------------------------
def test_get_series_ids():
    df = pd.DataFrame({"series_id": ["a", "a", "b", "c", "b"]})
    assert sorted(utils.get_series_ids(df)) == ["a", "b", "c"]


def test_get_series():
    df = pd.DataFrame({"series_id": ["a", "b", "a"], "step": [0, 1, 2]})
    out = utils.get_series(df, "a")
    assert set(out["step"]) == {0, 2}


# --- get_events --------------------------------------------------------------
def test_get_events_detects_transitions():
    # smoothed predictions: asleep(0) -> awake(1) is a wakeup; 1 -> 0 is onset.
    df = pd.DataFrame(
        {
            "series_id": ["s"] * 6,
            "step": [0, 1, 2, 3, 4, 5],
        }
    )
    y_pred = np.array([1, 1, 0, 0, 1, 1])  # onset at idx 2, wakeup at idx 4
    y_probas = [0.9] * 6
    out = utils.get_events(df, y_pred, y_probas)
    events = out.set_index("step")["event"]
    assert events[2] == "onset"
    assert events[4] == "wakeup"
    # untouched steps are NaN
    assert pd.isna(events[1])


# --- keep_periods ------------------------------------------------------------
def test_keep_periods_filters_short_sleep():
    # One long sleep period (>= min) and one tiny one that must be dropped.
    df = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "series_id": ["s", "s", "s", "s"],
            "step": [0, 100, 105, 110],
            "event": ["onset", "wakeup", "onset", "wakeup"],
            "score": [0.9, 0.9, 0.9, 0.9],
        }
    )
    out = utils.keep_periods(df, min_period=50)
    # First period (0->100) length 100 >= 50 kept; second (105->110) length 5 dropped.
    assert out["step"].tolist() == [0, 100]


def test_keep_periods_raises_without_complete_cycle():
    df = pd.DataFrame(
        {
            "row_id": [0],
            "series_id": ["s"],
            "step": [0],
            "event": ["onset"],  # no wakeup at all
            "score": [0.9],
        }
    )
    with pytest.raises((ValueError, IndexError)):
        utils.keep_periods(df, min_period=10)


# --- sleep metrics / per-night -----------------------------------------------
def _submission_two_nights():
    # Two clean onset->wakeup periods on one series, at 5s steps.
    # 0->720 (1h) and 2000->3440 (2h).
    return pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "series_id": ["s"] * 4,
            "step": [0, 720, 2000, 3440],
            "event": ["onset", "wakeup", "onset", "wakeup"],
            "score": [0.8, 0.9, 0.7, 0.95],
        }
    )


def _raw_for_submission():
    steps = list(range(0, 3460, 10))
    return pd.DataFrame(
        {
            "series_id": ["s"] * len(steps),
            "step": steps,
            "timestamp": pd.date_range("2024-01-01", periods=len(steps), freq="50s")
            .strftime("%Y-%m-%dT%H:%M:%S-0400"),
            "anglez": 0.0,
            "enmo": 0.1,
        }
    )


def test_compute_sleep_metrics():
    sub = _submission_two_nights()
    raw = _raw_for_submission()
    m = utils.compute_sleep_metrics(raw, sub, "s")
    assert m["periods"] == 2
    assert m["awakenings"] == 1  # 2 periods -> 1 gap
    # durations: 720 steps = 1h, 1440 steps = 2h
    assert m["longest_h"] == 2.0
    assert m["shortest_h"] == 1.0
    assert 0 <= m["efficiency"] <= 100


def test_per_night_breakdown():
    sub = _submission_two_nights()
    raw = _raw_for_submission()
    pn = utils.per_night_breakdown(raw, sub, "s")
    assert list(pn["night"]) == [1, 2]
    assert set(pn.columns) == {"night", "onset", "wakeup", "duration_h", "confidence"}
    assert pn.loc[0, "duration_h"] == 1.0
    assert pn.loc[1, "duration_h"] == 2.0


def test_batch_summary():
    sub = _submission_two_nights()
    raw = _raw_for_submission()
    table = utils.batch_summary(raw, sub)
    assert list(table["series_id"]) == ["s"]
    assert table.loc[0, "periods"] == 2


# --- synthetic generator -----------------------------------------------------
def test_generate_synthetic_series_schema_and_count():
    df = utils.generate_synthetic_series(n_series=2, days=2, seed=1)
    assert utils.check_df(df) is True
    assert df["series_id"].nunique() == 2
    # 2 days * 17280 steps/day * 2 series
    assert len(df) == 2 * (2 * 24 * 60 * 60 // utils.SECONDS_PER_STEP)


def test_generate_synthetic_series_is_reproducible():
    a = utils.generate_synthetic_series(n_series=1, days=2, seed=42)
    b = utils.generate_synthetic_series(n_series=1, days=2, seed=42)
    assert a["series_id"].iloc[0] == b["series_id"].iloc[0]
    assert a["enmo"].equals(b["enmo"])


# --- threshold behaviour -----------------------------------------------------
class _FakePipeline:
    """Minimal pipeline returning fixed positive-class probabilities."""

    def __init__(self, pos):
        self._pos = np.asarray(pos, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self._pos, self._pos])


def test_threshold_changes_predictions():
    df = pd.DataFrame({c: [0, 0, 0] for c in utils.MODEL_FEATURES})
    pipe = _FakePipeline([0.4, 0.6, 0.8])
    low, _ = utils.get_predictions_probas(df, pipe, threshold=0.3)
    high, _ = utils.get_predictions_probas(df, pipe, threshold=0.7)
    assert low.tolist() == [1, 1, 1]   # all >= 0.3
    assert high.tolist() == [0, 0, 1]  # only 0.8 >= 0.7


# --- chart helpers -----------------------------------------------------------
def test_subsample_caps_points():
    df = pd.DataFrame({"a": range(50000)})
    out = utils._subsample(df, max_points=1000)
    assert len(out) <= 1000
    # small frames are returned unchanged
    small = pd.DataFrame({"a": range(10)})
    assert len(utils._subsample(small, max_points=1000)) == 10


def test_plotly_prediction_downsamples_and_has_state_band():
    sub = _submission_two_nights()
    raw = _raw_for_submission()
    fig = utils.plotly_prediction(raw, sub, "s", max_points=50)
    enmo = [t for t in fig.data if t.name == "enmo"][0]
    assert len(enmo.x) <= 50
    assert any(t.name == "state" for t in fig.data)  # hypnogram band
    # anglez trace only appears when toggled
    assert not any(t.name == "anglez" for t in fig.data)
    fig2 = utils.plotly_prediction(raw, sub, "s", show_anglez=True, max_points=50)
    assert any(t.name == "anglez" for t in fig2.data)


def test_plotly_actogram_builds():
    sub = _submission_two_nights()
    raw = _raw_for_submission()
    fig = utils.plotly_actogram(raw, sub, "s")
    assert fig is not None
    assert len(fig.data) == 1  # one heatmap


# --- rate_metric -------------------------------------------------------------
@pytest.mark.parametrize(
    "kind,value,expected_label,expected_color",
    [
        ("efficiency", 90, "good", "normal"),
        ("efficiency", 80, "fair", "off"),
        ("efficiency", 50, "low", "inverse"),
        ("latency", 0.4, "good", "normal"),
        ("latency", 2.0, "long", "inverse"),
        ("awakenings", 0, "good", "normal"),
        ("awakenings", 5, "high", "inverse"),
        ("latency", None, "—", "off"),
    ],
)
def test_rate_metric(kind, value, expected_label, expected_color):
    label, color = utils.rate_metric(kind, value)
    assert label == expected_label
    assert color == expected_color


# --- recording_span ----------------------------------------------------------
def test_recording_span_with_timestamps():
    raw = _raw_for_submission()
    span = utils.recording_span(raw, "s")
    assert span["start"] is not None and span["end"] is not None
    assert span["days"] >= 1
    assert "day(s)" in span["label"]


def test_recording_span_without_timestamps():
    raw = _raw_for_submission().drop(columns=["timestamp"])
    span = utils.recording_span(raw, "s")
    assert span["start"] is None
    assert span["days"] >= 1
    assert "day(s)" in span["label"]
