# Architecture

This document explains how the **Detect Sleep States** app turns raw
accelerometer data into predicted sleep `onset` / `wakeup` events. It is the
deep-dive companion to the [README](../README.md).

All of the logic below lives in [`utils.py`](../utils.py); the Streamlit UI in
[`app.py`](../app.py) just orchestrates these steps.

---

## Pipeline overview

```
raw data ─▶ check_df ─▶ feature_engineering ─▶ pipeline_v2.pkl ─▶ smooth_results
                                                                      │
   metrics + charts ◀─ keep_periods ◀─ get_submission ◀─ get_events ◀─┘
```

| Step | Function | Purpose |
| --- | --- | --- |
| Validate | `check_df` | Ensure required columns and dtypes are present. |
| Features | `feature_engineering` | Build the 13 model inputs. |
| Predict | `get_predictions_probas` | Positive-class probability → class (at a tunable `threshold`) + confidence. |
| Smooth | `smooth_results` | Per-series rolling mean (tunable window), then re-binarise. |
| Events | `get_events` | Convert 0↔1 transitions into onset/wakeup. |
| Reduce | `get_submission` | Keep submission columns + `row_id`. |
| Filter | `keep_periods` | Drop periods shorter than the (tunable) minimum. |
| Summarise | `compute_sleep_metrics`, `per_night_breakdown`, `batch_summary`, `recording_span`, `rate_metric` | Clinical KPIs (with good/fair/low ratings), per-night table, multi-series rollup, recording date-range. |
| Visualise | `plotly_prediction`, `plotly_actogram`, `plotly_nightly_trend`, `plotly_feature_importance` | Interactive charts. `plotly_prediction` subsamples to `MAX_PLOT_POINTS` (≈6000) for fast rendering, draws a hypnogram state band, and can overlay `anglez`. |

> **Tunable settings.** `get_predictions_probas` takes a `threshold`, and
> `build_submission` takes `smooth_window` and `min_sleep_steps`. The app exposes
> all three as sidebar sliders; defaults reproduce the original behaviour.

---

## 1. Feature engineering

`feature_engineering(df)` runs two stages and drops helper columns
(`datetime`, `month`, `weekday`) at the end.

### `preprocess_col(df, "enmo", rolling_val)`

The rolling window adapts to the data: `rolling_val = rows / n_series / 10`
(≈ one tenth of the average series length). For each `series_id`:

1. **`rolling_mean`** — centered rolling mean of `enmo` over `rolling_val`
   steps, with `bfill().ffill()` to fill the centered-window edges.
2. **`high_to_mean`** — the rolling mean **clipped at the series mean** (values
   above the mean are pulled down to it). This flattens the high-activity
   (awake) regions so the low-activity (sleep) troughs stand out.
3. **`enmo_centered`** — `high_to_mean` shifted down by `|mean + std| / 2`,
   producing a roughly zero-centered sleep/wake signal.

Only `enmo_centered` is kept.

### `get_features(df)`

Adds, vectorised per the implementation:

| Feature | Definition |
| --- | --- |
| `hour` | `datetime.dt.hour` |
| `moment` | `get_moment(hour)` → night / morning / afternoon / evening |
| `anglez_abs` | `anglez.abs()` |
| `anglez_diff` | per-series `anglez_abs.diff(periods=12)`, back-filled |
| `enmo_diff` | per-series `enmo.diff(periods=12)`, back-filled |
| `anglez_rolling_mean` | centered rolling mean of `anglez_abs`, window 12 |
| `enmo_rolling_mean` | centered rolling mean of `enmo`, window 12 |
| `enmo_x_anglez` | `enmo * anglez` |
| `enmo_x_anglez_abs` | `enmo * anglez_abs` |
| `is_weekend` | `1` if weekday ≥ 5 else `0` |

`FEATURE_PERIODS = 12` steps = **1 minute** (records are 5 seconds apart).

### Final model inputs (`MODEL_FEATURES`)

```
anglez, enmo, enmo_centered, hour, moment, anglez_abs, anglez_diff,
enmo_diff, anglez_rolling_mean, enmo_rolling_mean, enmo_x_anglez,
enmo_x_anglez_abs, is_weekend
```

---

## 2. Model

The app serves **`pipeline_v2.pkl`**, a scikit-learn `Pipeline` with two steps:

| Step | Component | Role |
| --- | --- | --- |
| `prepross` | `ColumnTransformer` (`StandardScaler` on numeric features, `OneHotEncoder` on `moment`) | Scale numerics, one-hot encode the categorical. |
| `xgb` | `XGBClassifier` | Predict `P(awake)`. |

It predicts the binary `awake` target and exposes `predict` / `predict_proba`.

### How `pipeline_v2.pkl` is produced — self-distillation

The original `pipeline_01_legacy.pkl` was serialized with scikit-learn 1.3.x and
only unpickles cleanly on that version (newer scikit-learn corrupts its
`ColumnTransformer`, raising `'str' object has no attribute 'transform'`). To
remove that version lock, [`scripts/train_model.py`](../scripts/train_model.py)
**self-distills** it:

1. the legacy model labels the bundled sample series (`awake` = teacher labels);
2. a fresh pipeline is trained on those labels with whatever scikit-learn / XGBoost
   is installed, using **`GroupKFold` by `series_id`** so a subject's days never
   leak across folds;
3. the final model is fit on all data and saved as `pipeline_v2.pkl`, with metrics
   written to [`models/model_metrics.json`](../models/model_metrics.json).

The distilled model reproduces the original closely (out-of-fold average precision
≈ 0.999, ~98 % row agreement) and loads on **any** modern scikit-learn. The app
loads the first of `pipeline_v2.pkl` → `pipeline_01.pkl` → `pipeline_01_legacy.pkl`
that exists, so it still works (in a Python-3.11 env) if only the legacy pickle is
present. Run `python scripts/train_model.py --train-data PATH` to train on real
`awake` labels instead of distilling.

The **Model** tab in the app renders `model_metrics.json`: CV average precision,
OOF ROC-AUC / accuracy, and a feature-importance bar chart.

### Exploratory notebook (separate)

[`detect-sleep-states-starter-notebook-ensemble.ipynb`](../detect-sleep-states-starter-notebook-ensemble.ipynb)
explores a *different* approach — a **LightGBM + Random Forest** ensemble — and is
**not** what the app serves. See the train/serve discrepancy below.

---

## 3. Post-processing

Raw per-step predictions are noisy, so they are turned into a few clean events:

1. **`smooth_results(df, y_pred, SMOOTH_WINDOW=1000)`** — per-series centered
   rolling mean over 1000 steps (≈ 83 min), re-binarised at 0.5. This removes
   short flickers between awake/asleep.
2. **`get_events(df, y_smooth, y_probas)`** — diff the smoothed prediction per
   series: a `1 → 0` transition is an **`onset`** (falling asleep), `0 → 1` is a
   **`wakeup`**; everything else is `NaN`.
3. **`get_submission(df)`** — drop non-events, keep
   `series_id, step, event, score`, and add a `row_id` index.
4. **`keep_periods(df, MIN_SLEEP_STEPS=720)`** — enforce that each series starts
   with an onset and ends with a wakeup, then discard any sleep **or** activity
   period shorter than `MIN_SLEEP_STEPS` (12 × 60 = 720 steps = **1 hour**).

All thresholds are named constants at the top of `utils.py` (and are overridable
per call) so they can be tuned in one place — or live, via the app's sidebar.

---

## 4. Sleep metrics & reporting

On top of the raw events, `utils.py` derives the figures the app surfaces:

| Function | Output |
| --- | --- |
| `compute_sleep_summary` | periods, onsets/wakeups, total sleep, avg confidence. |
| `compute_sleep_metrics` | the above **plus** efficiency (% of recording span asleep), sleep latency (start → first onset), avg/longest/shortest period, awakenings. |
| `per_night_breakdown` | one row per sleep period with local onset/wakeup clock times, duration and confidence. |
| `batch_summary` | one summary row per `series_id` (for the Batch tab); skips series with no derivable period. |
| `build_html_report` | a self-contained HTML report (open in a browser, Print → Save as PDF). |
| `generate_synthetic_series` | day/night synthetic data in the app schema for testing the upload path. |

> **Note on efficiency.** It is `total sleep ÷ recording span`. Because the sample
> recordings include long awake daytime stretches, efficiency reads low (~15–25 %);
> the app labels it with a tooltip.

---

## 5. App structure

[`app.py`](../app.py) is a tabbed Streamlit app. The **sidebar** holds a
**🌗 dark/light theme toggle** (which calls `set_chart_theme` to switch the Plotly
template), the data source (sample / upload / **synthetic generator**), an inline
**data preview**, and the detection-setting sliders (threshold, smoothing window,
minimum sleep period) with a **reset-to-defaults** button.

> The small per-night and batch tables render with `st.table` (static) rather than
> `st.dataframe`; on Streamlit 1.28 the interactive data grid logs a harmless
> "Row index is out of range" console error when re-rendered with fewer rows.

The analysis (`feature_engineering` + prediction + post-processing) is run **once
per render** inside an `st.status` panel — so the main area shows progress instead
of a blank pane — and is cached with `@st.cache_data` keyed on a content
fingerprint. The resulting `ctx` is **shared across all tabs**, so switching tabs or
changing the visualised series does not recompute the model.

| Tab | Shows |
| --- | --- |
| 🔍 Analyze | 8 KPI cards (with good/fair/low ratings), recording date-range, interactive timeline (+ hypnogram band, optional `anglez` trace), per-night table + nightly trend, **actogram** heatmap, downloads (CSV + HTML report). |
| 📊 Batch | `batch_summary` table across every series. |
| 🆚 Compare | two series side by side. |
| 🤖 Model | model card from `model_metrics.json` + feature importances. |

An **About & how it works** expander (with the worked-example images) and a
branding footer sit below the tabs and are reachable from any state.

---

## 6. Train / serve discrepancy (important)

The exploratory notebook's `create_features()` and the app's
`feature_engineering()` were developed independently and differ:

| | Notebook | App (`utils.py`) |
| --- | --- | --- |
| Model | LightGBM + Random Forest ensemble | XGBoost pipeline (`pipeline_v2.pkl`) |
| Feature builder | `create_features` | `feature_engineering` |
| Rolling window | fixed `period = 50` | adaptive `rows / n_series / 10` |
| Feature count | 11 | 13 |
| Extra features | — | `enmo_centered`, `moment`, `is_weekend` |

The served model is the **distilled XGBoost pipeline** (`pipeline_v2.pkl`), trained
through `scripts/train_model.py` using the app's own `feature_engineering` — so the
**served model and the app are now feature-consistent** (train == serve for the
deployed path). The notebook remains a separate exploration and is not wired into
the app. Distillation inherits the legacy model's behaviour rather than ground
truth; training on real `awake` labels (`--train-data`) is the path to a model that
is independent of the legacy pickle.
