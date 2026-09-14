#!/usr/bin/env python3
"""Test whether conventional descriptors jointly explain pathway-model benefit.

This is an explanatory auxiliary experiment, not a barrier-prediction benchmark.
It uses one seed-averaged observation per LiTraj hop, repeated material-grouped
cross-validation, fold-local imputation/scaling, target-permutation nulls, and
held-out permutation importance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PUBLISHED = [
    "litraj_min_volume", "litraj_edge_length",
    "litraj_max_weighted_mean_oxidation", "litraj_max_volume",
    "litraj_min_mean_covalent_radius",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path(
        "results/litraj_classical_profile_analysis/per_hop_with_litraj_descriptors.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/multivariate_descriptor_experiment"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=10)
    p.add_argument("--permutations", type=int, default=100,
                   help="target permutations; use 1000 for the final dissertation run")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--rf-trees", type=int, default=200)
    p.add_argument("--rf-min-leaf", type=int, default=5)
    p.add_argument("--include-classification", action="store_true",
                   help="also predict whether improvement is positive")
    p.add_argument("--outcomes", nargs="+", choices=["delta_pos", "delta_bvse"],
                   default=["delta_pos", "delta_bvse"])
    return p


def material_id(source: str) -> str:
    match = re.search(r"(mp-\d+)", str(source))
    return match.group(1) if match else str(source)


def profile_features(prefix: str) -> list[str]:
    properties = ("volume", "effective_cn", "mean_rc", "weighted_mean_oxi")
    summaries = ("min", "max", "range", "s_min", "s_max", "asymmetry")
    return [f"{prefix}_profile_{prop}_{summary}"
            for prop in properties for summary in summaries]


def feature_sets(outcome: str) -> dict[str, list[str]]:
    # Q3 uses the DFT path. Q4 deliberately uses BVSE descriptors only, avoiding
    # DFT-path information that would not exist in the cheap-input setting.
    prefix = "dft" if outcome == "delta_pos" else "bvse"
    geometry = [
        f"{prefix}_path_length", f"{prefix}_tortuosity",
        f"{prefix}_bottleneck_mean_k_distance", f"{prefix}_coordination_range",
    ]
    profiles = profile_features(prefix)
    return {
        "published_litraj": PUBLISHED,
        "simple_geometry": geometry,
        "trajectory_profiles": profiles,
        "published_plus_profiles": PUBLISHED + profiles,
    }


def regression_models(args, seed: int):
    ridge = TransformedTargetRegressor(
        regressor=Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=args.ridge_alpha)),
        ]),
        transformer=StandardScaler(),
    )
    forest = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", RandomForestRegressor(
            n_estimators=args.rf_trees, min_samples_leaf=args.rf_min_leaf,
            max_features="sqrt", random_state=seed, n_jobs=1,
        )),
    ])
    return {"ridge": ridge, "random_forest": forest}


def classification_models(args, seed: int):
    logistic = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
    ])
    forest = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", RandomForestClassifier(
            n_estimators=args.rf_trees, min_samples_leaf=args.rf_min_leaf,
            max_features="sqrt", random_state=seed, class_weight="balanced",
            n_jobs=1,
        )),
    ])
    return {"logistic": logistic, "random_forest": forest}


def splits(groups: np.ndarray, args, repeat: int):
    splitter = GroupKFold(n_splits=args.folds, shuffle=True,
                          random_state=args.seed + repeat)
    dummy = np.zeros(len(groups))
    return list(splitter.split(dummy, groups=groups))


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    rho = spearmanr(y, prediction).statistic if np.ptp(prediction) else np.nan
    return {"r2": r2_score(y, prediction),
            "mae": mean_absolute_error(y, prediction), "spearman": rho}


def classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = probability >= .5
    return {"roc_auc": safe_auc(y, probability),
            "balanced_accuracy": balanced_accuracy_score(y, predicted)}


def safe_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return roc_auc_score(y, probability) if len(np.unique(y)) == 2 else np.nan


def run_cv(data: pd.DataFrame, outcome: str, set_name: str, features: list[str],
           task: str, args, permuted_y: np.ndarray | None = None,
           collect_importance: bool = False):
    X = data[features].to_numpy(float)
    raw_y = data[outcome].to_numpy(float)
    observed_y = raw_y if task == "regression" else (raw_y > 0).astype(int)
    y = observed_y if permuted_y is None else permuted_y
    groups = data.material_id.to_numpy()
    prediction_rows, metric_rows, importance_rows = [], [], []
    for repeat in range(args.cv_repeats):
        for fold, (train, validation) in enumerate(splits(groups, args, repeat)):
            seed = args.seed + repeat * 100 + fold
            models = (regression_models(args, seed) if task == "regression"
                      else classification_models(args, seed))
            for model_name, model in models.items():
                model.fit(X[train], y[train])
                if task == "regression":
                    predicted = model.predict(X[validation])
                else:
                    predicted = model.predict_proba(X[validation])[:, 1]
                for index, value in zip(validation, predicted):
                    prediction_rows.append({
                        "outcome": outcome, "feature_set": set_name, "task": task,
                        "model": model_name, "repeat": repeat, "fold": fold,
                        "source": data.iloc[index].source, "material_id": groups[index],
                        "target": raw_y[index], "fit_target": y[index], "prediction": value,
                    })
                if collect_importance and model_name == "random_forest":
                    baseline = (r2_score(y[validation], predicted) if task == "regression"
                                else safe_auc(y[validation], predicted))
                    rng = np.random.default_rng(seed + 500_000)
                    for column, feature in enumerate(features):
                        changed = X[validation].copy()
                        changed[:, column] = rng.permutation(changed[:, column])
                        changed_prediction = (model.predict(changed) if task == "regression"
                                              else model.predict_proba(changed)[:, 1])
                        score = (r2_score(y[validation], changed_prediction)
                                 if task == "regression"
                                 else safe_auc(y[validation], changed_prediction))
                        importance_rows.append({
                            "outcome": outcome, "feature_set": set_name, "task": task,
                            "repeat": repeat, "fold": fold, "feature": feature,
                            "score_drop": baseline - score,
                        })
        repeat_predictions = pd.DataFrame(prediction_rows)
        repeat_predictions = repeat_predictions[repeat_predictions.repeat.eq(repeat)]
        for model_name, model_rows in repeat_predictions.groupby("model"):
            ordered = model_rows.set_index("source").loc[data.source]
            values = ordered.prediction.to_numpy(float)
            metrics = (regression_metrics(y, values) if task == "regression"
                       else classification_metrics(y, values))
            metric_rows.append({"outcome": outcome, "feature_set": set_name,
                                "task": task, "model": model_name, "repeat": repeat,
                                **metrics})
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows), pd.DataFrame(importance_rows)


def primary_score(metrics: pd.DataFrame, task: str) -> float:
    column = "r2" if task == "regression" else "roc_auc"
    return float(metrics[column].mean())


def experiment(data: pd.DataFrame, args):
    all_predictions, all_metrics, all_importances, null_rows = [], [], [], []
    rng = np.random.default_rng(args.seed + 900_000)
    tasks = ["regression"] + (["classification"] if args.include_classification else [])
    for outcome in args.outcomes:
        for set_name, features in feature_sets(outcome).items():
            missing = sorted(set(features) - set(data.columns))
            if missing:
                raise ValueError(f"{set_name} is missing columns: {missing}")
            for task in tasks:
                print(f"[{outcome}/{set_name}/{task}] observed CV", flush=True)
                predictions, metrics, importances = run_cv(
                    data, outcome, set_name, features, task, args,
                    collect_importance=True)
                all_predictions.append(predictions)
                all_metrics.append(metrics)
                all_importances.append(importances)
                for permutation in range(args.permutations):
                    raw = data[outcome].to_numpy(float)
                    permuted = rng.permutation(raw)
                    if task == "classification":
                        permuted = (permuted > 0).astype(int)
                    _, null_metrics, _ = run_cv(
                        data, outcome, set_name, features, task, args,
                        permuted_y=permuted, collect_importance=False)
                    for model, rows in null_metrics.groupby("model"):
                        null_rows.append({
                            "outcome": outcome, "feature_set": set_name, "task": task,
                            "model": model, "permutation": permutation,
                            "score": primary_score(rows, task),
                        })
                    if (permutation + 1) % 10 == 0:
                        print(f"  permutation {permutation + 1}/{args.permutations}", flush=True)
    null_columns = ["outcome", "feature_set", "task", "model", "permutation", "score"]
    return (pd.concat(all_predictions, ignore_index=True),
            pd.concat(all_metrics, ignore_index=True),
            pd.concat(all_importances, ignore_index=True),
            pd.DataFrame(null_rows, columns=null_columns))


def summarise(metrics: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["outcome", "feature_set", "task", "model"]
    for key, group in metrics.groupby(keys):
        outcome, feature_set, task, model = key
        score_name = "r2" if task == "regression" else "roc_auc"
        observed = float(group[score_name].mean())
        matching = nulls[
            nulls.outcome.eq(outcome) & nulls.feature_set.eq(feature_set) &
            nulls.task.eq(task) & nulls.model.eq(model)
        ].score.to_numpy(float)
        row = dict(zip(keys, key))
        for column in group.columns:
            if column in keys or column == "repeat":
                continue
            row[f"{column}_mean"] = group[column].mean()
            row[f"{column}_sd"] = group[column].std(ddof=1)
        row["permutation_p"] = ((1 + np.sum(matching >= observed)) / (len(matching) + 1)
                                if len(matching) else np.nan)
        row["null_score_mean"] = matching.mean() if len(matching) else np.nan
        row["null_score_95"] = np.quantile(matching, .95) if len(matching) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_plot(summary: pd.DataFrame, output: Path) -> None:
    cache = output / ".matplotlib_cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    import matplotlib.pyplot as plt
    regression = summary[summary.task.eq("regression")]
    outcomes = regression.outcome.unique()
    fig, axes = plt.subplots(1, len(outcomes), figsize=(6 * len(outcomes), 4.5), squeeze=False)
    for ax, outcome in zip(axes.flat, outcomes):
        subset = regression[regression.outcome.eq(outcome)].copy()
        labels = subset.feature_set + "\n" + subset.model
        x = np.arange(len(subset))
        ax.axhline(0, color="0.5", lw=1)
        ax.errorbar(x, subset.r2_mean, yerr=subset.r2_sd, fmt="o", capsize=3)
        ax.scatter(x, subset.null_score_95, marker="_", s=180, label="null 95th percentile")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set(ylabel="Repeated grouped-CV $R^2$", title=outcome)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "multivariate_explanatory_results.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    if args.folds < 2 or args.cv_repeats < 1 or args.permutations < 0:
        raise ValueError("folds >= 2, cv-repeats >= 1, and permutations >= 0 are required")
    data = pd.read_csv(args.input)
    data["material_id"] = data.source.map(material_id)
    if data.source.duplicated().any():
        raise ValueError("input must contain one row per hop")
    if data.material_id.nunique() < args.folds:
        raise ValueError("fewer material groups than CV folds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    definitions = {outcome: feature_sets(outcome) for outcome in args.outcomes}
    (args.output_dir / "feature_sets.json").write_text(json.dumps(definitions, indent=2) + "\n")
    predictions, metrics, importances, nulls = experiment(data, args)
    summary = summarise(metrics, nulls)
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics_by_repeat.csv", index=False)
    importances.to_csv(args.output_dir / "heldout_permutation_importance.csv", index=False)
    nulls.to_csv(args.output_dir / "permutation_null_scores.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    make_plot(summary, args.output_dir)
    print("\n", summary.to_string(index=False), sep="")
    print(f"\nWrote outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
