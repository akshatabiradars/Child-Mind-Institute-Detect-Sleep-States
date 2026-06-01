"""Train (or re-train) the sleep/wake model used by the app.

By default this *self-distills* the bundled legacy pipeline: the legacy model
(``pipeline_01_legacy.pkl``, serialized with scikit-learn 1.3.x) is used to label
the sample series, and a fresh scikit-learn ``Pipeline`` is trained to reproduce
its behaviour. The result (``pipeline_v2.pkl``) is built with whatever
scikit-learn/XGBoost is installed, so it loads cleanly without the cross-version
``ColumnTransformer`` corruption that affects the legacy pickle.

If a labelled training set is available (a parquet/CSV with an ``awake`` column),
pass ``--train-data PATH`` to train on the real labels instead of distilling.

Usage::

    python scripts/train_model.py                 # self-distill from legacy pkl
    python scripts/train_model.py --train-data Zzzs_train.parquet
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Make the project importable when run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import MODEL_FEATURES, feature_engineering  # noqa: E402

LEGACY_PKL = ROOT / "pipeline_01_legacy.pkl"
OUT_PKL = ROOT / "pipeline_v2.pkl"
METRICS_JSON = ROOT / "models" / "model_metrics.json"
SAMPLE_FILES = [
    ROOT / "data" / "input_series_1.csv",
    ROOT / "data" / "input_series_2.csv",
    ROOT / "data" / "input_series_3.csv",
]

CATEGORICAL = ["moment"]
NUMERIC = [c for c in MODEL_FEATURES if c not in CATEGORICAL]


def build_pipeline(random_state: int = 42) -> Pipeline:
    """A modern, self-contained preprocessing + XGBoost pipeline."""
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="drop",
    )
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("prepross", pre), ("xgb", clf)])


def load_training_frame(train_data: str | None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (features_df, target, groups) for training.

    Distillation path: label the sample series with the legacy model.
    Real-label path: read ``--train-data`` and use its ``awake`` column.
    """
    if train_data:
        raw = (
            pd.read_parquet(train_data)
            if train_data.endswith(".parquet")
            else pd.read_csv(train_data)
        )
        if "awake" not in raw.columns:
            raise SystemExit("ERROR: --train-data must contain an 'awake' column.")
        prepared = feature_engineering(raw)
        y = raw["awake"].astype(int).to_numpy()
        print(f"Training on REAL labels: {len(prepared):,} rows from {train_data}")
        return prepared, pd.Series(y), prepared["series_id"]

    # Self-distillation.
    if not LEGACY_PKL.exists():
        raise SystemExit(f"ERROR: {LEGACY_PKL.name} not found (needed to distil).")
    with open(LEGACY_PKL, "rb") as f:
        legacy = pickle.load(f)

    frames = [pd.read_csv(p) for p in SAMPLE_FILES if p.exists()]
    if not frames:
        raise SystemExit("ERROR: no sample CSVs found in data/.")
    raw = pd.concat(frames, ignore_index=True)
    prepared = feature_engineering(raw)

    # Legacy teacher labels (1 = awake).
    y = legacy.predict(prepared[MODEL_FEATURES])
    y = np.asarray(y).astype(int)
    print(f"Self-distilling: {len(prepared):,} rows, teacher positive rate {y.mean():.3f}")
    return prepared, pd.Series(y), prepared["series_id"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-data", default=None, help="Optional labelled parquet/CSV with an 'awake' column.")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    prepared, y, groups = load_training_frame(args.train_data)
    X = prepared[MODEL_FEATURES]

    # Grouped CV by series_id so a subject's days never leak across folds.
    n_groups = groups.nunique()
    n_splits = min(5, max(2, n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    oof = np.zeros(len(y), dtype=float)
    fold_scores = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        model = build_pipeline(args.random_state)
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = proba
        ap_score = average_precision_score(y.iloc[va], proba)
        fold_scores.append(ap_score)
        print(f"  fold {fold + 1}/{n_splits}: AP={ap_score:.4f}")

    # Final model on all data.
    final = build_pipeline(args.random_state)
    final.fit(X, y)

    metrics = {
        "n_rows": int(len(y)),
        "n_series": int(n_groups),
        "positive_rate": float(y.mean()),
        "cv_folds": n_splits,
        "cv_average_precision_mean": float(np.mean(fold_scores)),
        "cv_average_precision_std": float(np.std(fold_scores)),
        "oof_average_precision": float(average_precision_score(y, oof)),
        "oof_roc_auc": float(roc_auc_score(y, oof)),
        "oof_accuracy_at_0.5": float(accuracy_score(y, (oof >= 0.5).astype(int))),
        "feature_order": MODEL_FEATURES,
        "trained_via": "real_labels" if args.train_data else "self_distillation",
    }

    # Feature importances (numeric + one-hot moment categories).
    try:
        ohe = final.named_steps["prepross"].named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL))
        feat_names = NUMERIC + cat_names
        importances = final.named_steps["xgb"].feature_importances_.tolist()
        metrics["feature_importances"] = dict(
            sorted(zip(feat_names, importances, strict=False), key=lambda kv: -kv[1])
        )
    except Exception as exc:  # pragma: no cover - importances are best-effort
        metrics["feature_importances_error"] = str(exc)

    with open(OUT_PKL, "wb") as f:
        pickle.dump(final, f)
    METRICS_JSON.parent.mkdir(exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved {OUT_PKL.name} and {METRICS_JSON.relative_to(ROOT)}")
    print(f"OOF AP={metrics['oof_average_precision']:.4f} "
          f"ROC-AUC={metrics['oof_roc_auc']:.4f} "
          f"acc@0.5={metrics['oof_accuracy_at_0.5']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
