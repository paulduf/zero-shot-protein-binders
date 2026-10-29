"""Command line entry point: build features, score, benchmark, report, plot."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from . import baselines, evaluate, figures, jev
from .data import LABEL_COLUMNS, build_feature_table, stratified_sample

CACHE_DIR = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


def features_path(label: str) -> Path:
    return CACHE_DIR / f"features_{label}.parquet"


def get_features(label: str, refresh: bool = False) -> pd.DataFrame:
    """Feature table for a label, cached to parquet because the download is slow."""
    path = features_path(label)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = build_feature_table(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def default_scored_path(label: str, model: str) -> Path:
    return RESULTS_DIR / f"scored_{label}_{model.split('/')[-1]}.csv"


def cmd_features(args: argparse.Namespace) -> None:
    df = get_features(args.label, refresh=args.refresh)
    counts = df["y_true"].value_counts()
    print(f"{args.label}: {len(df)} designs, {df.shape[1] - 2} features, {df['target'].nunique()} targets")
    print(counts.to_string())
    print(f"cached at {features_path(args.label)}")


def cmd_models(args: argparse.Namespace) -> None:
    print("\n".join(jev.list_models()))


def cmd_score(args: argparse.Namespace) -> None:
    df = get_features(args.label)
    if args.limit:
        df = stratified_sample(df, args.limit, args.seed)

    out_path = Path(args.out) if args.out else default_scored_path(args.label, args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    previous = pd.DataFrame()
    if args.resume and out_path.exists():
        previous = pd.read_csv(out_path)
        previous = previous[previous["p_binder"].notna()]
        print(f"resuming: {len(previous)} designs already scored in {out_path}")

    client = jev.JevClient(
        model=args.model,
        api_url=jev.PROD_API_URL if args.api_key_env else jev.DEMO_API_URL,
        api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
    )
    print(f"scoring {len(df) - len(previous)} designs with {args.model} ...")
    scored = jev.score_dataframe(
        df, client, rps=args.rps, workers=args.workers, already_scored=previous.get("uuid", [])
    )
    combined = pd.concat([previous, scored], ignore_index=True) if len(previous) else scored
    combined.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    print(evaluate.metrics_frame(evaluate.evaluate_scored(combined, args.model)).to_string())


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Rank the zero-shot run against supervised and single-feature baselines."""
    scored = pd.read_csv(args.scored).dropna(subset=["p_binder"])
    y_true = evaluate.binarize(scored["y_true"])

    candidates = {"zero-shot Jev": scored["p_binder"].to_numpy(dtype=float)}
    candidates.update(baselines.supervised_scores(
        scored, group_by_target=args.by_target, seed=args.seed, foundation_models=tuple(args.tfm)
    ))
    candidates.update(baselines.single_feature_scores(scored))

    rows = {}
    for name, scores in candidates.items():
        rows[f"{name} @0.5"] = evaluate.metrics(y_true, scores, 0.5)
        rows[f"{name} @best-f1"] = evaluate.metrics(y_true, scores, evaluate.best_f1_threshold(y_true, scores))

    table = evaluate.metrics_frame(rows).sort_values("roc_auc", ascending=False)
    out_path = Path(args.out) if args.out else Path(args.scored).with_name(
        Path(args.scored).stem.replace("scored", "benchmark") + ".csv"
    )
    table.to_csv(out_path)
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out_path}")


def cmd_report(args: argparse.Namespace) -> None:
    scored = pd.read_csv(args.scored)
    print(evaluate.metrics_frame(evaluate.evaluate_scored(scored)).to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print(evaluate.report(scored, args.threshold))


def cmd_figures(args: argparse.Namespace) -> None:
    runs = {Path(p).stem.replace("scored_", ""): pd.read_csv(p) for p in args.scored}
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    written = figures.render_all(runs, FIGURES_DIR)
    print("\n".join(str(p) for p in written))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="zsbinders", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    features = sub.add_parser("features", help="build and cache the feature table")
    features.add_argument("--label", choices=LABEL_COLUMNS, default="adaptyv")
    features.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    features.set_defaults(func=cmd_features)

    models = sub.add_parser("models", help="list the classifier models the demo API serves")
    models.set_defaults(func=cmd_models)

    score = sub.add_parser("score", help="score designs with the Simple Jev classifier API")
    score.add_argument("--label", choices=LABEL_COLUMNS, default="adaptyv")
    score.add_argument("--model", default=jev.DEFAULT_MODEL)
    score.add_argument("--limit", type=int, default=0, help="class-balanced subsample size (0 = all)")
    score.add_argument("--rps", type=float, default=jev.DEMO_RATE_LIMIT_RPS)
    score.add_argument("--workers", type=int, default=8)
    score.add_argument("--seed", type=int, default=1337)
    score.add_argument("--resume", action="store_true", help="skip designs already present in --out")
    score.add_argument("--api-key-env", help="env var holding a Featherless key; switches to the production API")
    score.add_argument("--out")
    score.set_defaults(func=cmd_score)

    benchmark = sub.add_parser("benchmark", help="compare a scored run against trained baselines")
    benchmark.add_argument("scored")
    benchmark.add_argument("--by-target", action="store_true", help="hold out whole antigens in CV")
    benchmark.add_argument("--seed", type=int, default=1337)
    benchmark.add_argument(
        "--tfm", action="append", default=[], choices=baselines.FOUNDATION_MODELS,
        help="also score a tabular foundation model (repeatable; needs the [tfm] extra)",
    )
    benchmark.add_argument("--out")
    benchmark.set_defaults(func=cmd_benchmark)

    report = sub.add_parser("report", help="per-class and per-target breakdown of a scored run")
    report.add_argument("scored")
    report.add_argument("--threshold", type=float, default=0.5)
    report.set_defaults(func=cmd_report)

    figs = sub.add_parser("figures", help="ROC, score distributions, and metric scatter plots")
    figs.add_argument("scored", nargs="+")
    figs.set_defaults(func=cmd_figures)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
