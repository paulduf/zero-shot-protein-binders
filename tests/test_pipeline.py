"""Offline checks on the pieces that would silently corrupt a run."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from zsbinders import baselines, evaluate, jev
from zsbinders.data import feature_columns, stratified_sample


@pytest.fixture
def scored() -> pd.DataFrame:
    """A separable toy run: p_binder tracks the label, offset so 0.5 is the wrong cut."""
    rng = np.random.default_rng(0)
    n = 120
    y_true = np.where(np.arange(n) % 3 == 0, "binder", "non_binder")
    signal = np.where(y_true == "binder", 0.9, 0.6) + rng.normal(0, 0.02, n)
    return pd.DataFrame(
        {
            "uuid": [f"d{i}" for i in range(n)],
            "target": np.where(np.arange(n) % 2 == 0, "antigen_a", "antigen_b"),
            "generator": "gen",
            "sequence_design_method": "method",
            "binder_length": rng.integers(50, 120, n),
            "epitope_n_residues": rng.integers(5, 30, n),
            "ipsae_min_mean": signal,
            "sc_dockq_mean": signal * 0.5,
            "y_true": y_true,
            "p_binder": np.clip(signal, 0, 1),
            "y_pred": np.where(signal >= 0.5, "binder", "non_binder"),
        }
    )


def test_feature_columns_exclude_labels_and_scoring_output(scored):
    columns = feature_columns(scored)
    assert set(columns) == {"binder_length", "epitope_n_residues", "ipsae_min_mean", "sc_dockq_mean"}


def test_stratified_sample_is_balanced_and_reproducible(scored):
    first = stratified_sample(scored, 20, seed=7)
    assert first["y_true"].value_counts().nunique() == 1
    assert len(first) == 20
    pd.testing.assert_frame_equal(first, stratified_sample(scored, 20, seed=7))


def test_tuned_threshold_recovers_the_separation_that_0_5_misses(scored):
    """The whole point of reporting a tuned cut: 0.5 scores at chance on separable scores."""
    results = evaluate.evaluate_scored(scored)
    at_half = results["zero-shot @0.5"]
    tuned = results["zero-shot @best-f1"]

    assert at_half["recall"] == 1.0 and at_half["balanced_accuracy"] == pytest.approx(0.5)
    assert tuned["balanced_accuracy"] > 0.95
    assert at_half["roc_auc"] == tuned["roc_auc"] > 0.99


def test_failed_rows_are_dropped_not_counted_as_predictions(scored):
    scored.loc[:9, ["p_binder", "y_pred"]] = None
    assert evaluate.evaluate_scored(scored)["zero-shot @0.5"]["n"] == len(scored) - 10


def test_baselines_separate_the_toy_signal(scored):
    supervised = baselines.supervised_scores(scored, seed=0)
    y_true = evaluate.binarize(scored["y_true"])
    for name, scores in supervised.items():
        assert evaluate.metrics(y_true, scores)["roc_auc"] > 0.9, name

    single = baselines.single_feature_scores(scored)
    assert set(single) == {"feature: ipsae_min_mean", "feature: sc_dockq_mean"}
    assert all(((s >= 0) & (s <= 1)).all() for s in single.values())


def test_single_feature_sign_flips_metrics_where_lower_is_better():
    """pae_interface is a distance-like error term, so it must be negated before scoring."""
    df = pd.DataFrame({"pae_interface_mean_mean": [1.0, 5.0, 9.0]})
    scores = baselines.single_feature_scores(df)["feature: pae_interface_mean_mean"]
    assert scores[0] > scores[1] > scores[2]


def test_request_body_carries_features_but_never_the_label(scored):
    payload = jev.JevClient().build_request(scored.iloc[0])
    assert payload["questions"]["binding"]["type"] == "noul"
    assert "ipsae_min_mean" in payload["state"]
    assert not {"uuid", "y_true"} & set(payload["state"])


def test_rate_limiter_holds_the_aggregate_rate():
    limiter = jev.RateLimiter(rps=20)
    start = time.monotonic()
    for _ in range(10):
        limiter.acquire()
    assert time.monotonic() - start >= 0.45


def test_foundation_models_score_on_the_same_folds(scored, monkeypatch):
    """A registered foundation model gets the raw frame, string columns included, and no label."""
    seen_columns = []

    def stand_in(seed):
        from sklearn.linear_model import LogisticRegression

        class Recording(baselines.Pipeline):
            def fit(self, X, y):
                seen_columns.append(list(X.columns))
                return super().fit(X, y)

        numeric = feature_columns(scored)
        steps = baselines.make_pipeline(LogisticRegression(), numeric, baselines.CATEGORICAL_COLUMNS).steps
        return Recording(steps)

    monkeypatch.setitem(baselines.FOUNDATION_MODELS, "stand-in", ("stand-in", stand_in))
    for by_target in (False, True):
        seen_columns.clear()
        scores = baselines.supervised_scores(scored, group_by_target=by_target, seed=0, foundation_models=("stand-in",))
        assert evaluate.metrics(evaluate.binarize(scored["y_true"]), scores["stand-in"])["roc_auc"] > 0.9
        assert seen_columns and all("target" in c and not {"y_true", "p_binder", "uuid"} & set(c) for c in seen_columns)


def test_tabicl_separates_the_toy_signal(scored):
    """Downloads the TabICLv2 checkpoint on first run."""
    pytest.importorskip("tabicl", reason="needs the [tfm] extra")
    scores = baselines.supervised_scores(scored, seed=0, foundation_models=("tabicl",))["TabICLv2"]
    assert evaluate.metrics(evaluate.binarize(scored["y_true"]), scores)["roc_auc"] > 0.9
