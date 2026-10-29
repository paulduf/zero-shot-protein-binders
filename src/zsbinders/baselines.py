"""Supervised and single-feature baselines on the same designs the zero-shot run scored.

The zero-shot result only means something next to the alternatives available to anyone
holding this feature table: a trained classifier, and the individual cofolding metrics used
as scores on their own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import feature_columns
from .evaluate import binarize

CATEGORICAL_COLUMNS = ["target", "generator", "sequence_design_method"]

# Scored directly as p(binder) with no training. Signs are set so higher means more likely
# to bind: the DockQ RMSD metrics and interface PAE are better when lower.
SINGLE_FEATURE_SCORES = {
    "ipsae_min_mean": 1,
    "ipsae_min_max": 1,
    "sc_dockq_mean": 1,
    "sc_dockq_max": 1,
    "iptm_pae_mean": 1,
    "plddt_binder_mean": 1,
    "n_interface_contacts_mean": 1,
    "pae_interface_mean_mean": -1,
}


def tabicl_classifier(seed: int):
    """TabICLv2, an in-context tabular foundation model: no gradient training per fold.

    It ordinal-encodes string columns itself and maps an antigen unseen in the context to
    its own code, so it takes the raw feature frame rather than the one-hot pipeline.
    """
    from tabicl import TabICLClassifier

    return TabICLClassifier(random_state=seed)


# Optional baselines that need extra dependencies (`pip install -e ".[tfm]"`) and model
# weights, so they only run when asked for by name.
FOUNDATION_MODELS = {"tabicl": ("TabICLv2", tabicl_classifier)}


def make_pipeline(estimator, numeric: list[str], categorical: list[str]) -> Pipeline:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if isinstance(estimator, LogisticRegression):
        numeric_steps.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical),
        ]
    )
    return Pipeline([("prep", preprocess), ("model", estimator)])


def supervised_scores(
    df: pd.DataFrame,
    group_by_target: bool = False,
    seed: int = 1337,
    foundation_models: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Out-of-fold p(binder) for each trained baseline.

    With `group_by_target`, folds hold out whole antigens, which measures transfer to a
    target the model never trained on. Stratified k-fold instead lets a model memorise
    per-target base rates, so it reads high.
    """
    y = binarize(df["y_true"])
    numeric = feature_columns(df)
    categorical = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    if group_by_target:
        splitter, groups = LeaveOneGroupOut(), df["target"]
        # A target with a single observed class contributes no usable held-out fold.
        usable = df.groupby("target")["y_true"].nunique() > 1
        if not usable.all():
            print(f"[warn] single-class targets excluded from grouped CV: {list(usable[~usable].index)}")
    else:
        splitter, groups = StratifiedKFold(5, shuffle=True, random_state=seed), None

    estimators = {
        "logistic regression": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "gradient boosting": HistGradientBoostingClassifier(random_state=seed),
    }
    scores = {}
    for name, estimator in estimators.items():
        pipeline = make_pipeline(estimator, numeric, categorical)
        probabilities = cross_val_predict(
            pipeline, df, y, cv=splitter, groups=groups, method="predict_proba"
        )
        scores[name] = probabilities[:, 1]

    for key in foundation_models:
        name, factory = FOUNDATION_MODELS[key]
        probabilities = cross_val_predict(
            factory(seed), df[numeric + categorical], y, cv=splitter, groups=groups, method="predict_proba"
        )
        scores[name] = probabilities[:, 1]
    return scores


def single_feature_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each raw metric as its own classifier, rescaled to [0, 1] so thresholds are comparable."""
    scores = {}
    for column, sign in SINGLE_FEATURE_SCORES.items():
        if column not in df.columns:
            continue
        values = sign * df[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        low, high = finite.min(), finite.max()
        span = high - low
        normalized = np.full_like(values, 0.5) if span == 0 else (values - low) / span
        scores[f"feature: {column}"] = np.nan_to_num(normalized, nan=0.5)
    return scores
