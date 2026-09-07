#!/usr/bin/env python3
"""No-retraining checks: split-seed benefit reliability and ensemble transparency."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path,
                   default=Path("../results/final_statistics/per_sample_predictions.csv"))
    p.add_argument("--fold-seed-metrics", type=Path,
                   default=Path("results/material_disjoint_evaluation/fold_seed_metrics.csv"))
    p.add_argument("--fold-ensemble-metrics", type=Path,
                   default=Path("results/material_disjoint_evaluation/fold_metrics.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("../results/robustness_reporting"))
    return p


def read(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranked


def correlations(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    return float(np.corrcoef(a, b)[0, 1]), float(np.corrcoef(rank(a), rank(b))[0, 1])


def benefit(rows, left, right, seeds) -> dict[str, float]:
    lookup = {(r["scenario"], r["method"], int(r["seed"]), r["source"]):
              float(r["absolute_error"]) for r in rows}
    sources = sorted({key[3] for key in lookup if key[:2] == left})
    result = {}
    for source in sources:
        values = []
        for seed in seeds:
            lk = (*left, seed, source); rk = (*right, seed, source)
            if lk in lookup and rk in lookup:
                values.append(lookup[lk] - lookup[rk])
        if values:
            result[source] = float(np.mean(values))
    return result


def reliability_rows(rows) -> list[dict]:
    comparisons = {
        "positional_benefit": (("dft_path", "midpoint"), ("dft_path", "positional")),
        "bvse_delta_benefit": (("affine_bvse", "affine_em_bvse"),
                               ("bvse_calibrated_delta", "positional")),
    }
    all_seeds = sorted({int(r["seed"]) for r in rows})
    if len(all_seeds) < 4:
        raise ValueError("at least four prediction seeds are required")
    half = len(all_seeds) // 2
    output = []
    lookup = {(r["scenario"], r["method"], int(r["seed"]), r["source"]):
              float(r["absolute_error"]) for r in rows}
    for name, (left, right) in comparisons.items():
        sources = sorted({key[3] for key in lookup if key[:2] == left})
        complete_sources = [source for source in sources if all(
            (*left, run_seed, source) in lookup and (*right, run_seed, source) in lookup
            for run_seed in all_seeds
        )]
        differences = np.asarray([
            [lookup[(*left, run_seed, source)] - lookup[(*right, run_seed, source)]
             for source in complete_sources]
            for run_seed in all_seeds
        ])
        # A complementary A/B split is equivalent to B/A for correlation, so
        # fix the first seed in A and enumerate each unique partition once.
        for repeat, remainder in enumerate(itertools.combinations(range(1, len(all_seeds)), half - 1)):
            indices_a = np.asarray((0, *remainder), dtype=int)
            indices_b = np.asarray(sorted(set(range(len(all_seeds))) - set(indices_a)), dtype=int)
            if differences.shape[1] < 3:
                continue
            pearson, spearman = correlations(differences[indices_a].mean(0),
                                             differences[indices_b].mean(0))
            output.append({"outcome": name, "repeat": repeat,
                           "n_hops": len(complete_sources),
                           "seeds_a": ";".join(map(str, sorted(all_seeds[i] for i in indices_a))),
                           "seeds_b": ";".join(map(str, sorted(all_seeds[i] for i in indices_b))),
                           "pearson": pearson, "spearman": spearman})
    return output


def reliability_summary(rows: list[dict]) -> list[dict]:
    output = []
    for outcome in sorted({r["outcome"] for r in rows}):
        subset = [r for r in rows if r["outcome"] == outcome]
        row = {"outcome": outcome, "unique_complementary_splits": len(subset),
               "n_hops": subset[0]["n_hops"]}
        for metric in ("pearson", "spearman"):
            values = np.asarray([float(r[metric]) for r in subset])
            row[f"median_{metric}"] = np.median(values)
            row[f"q025_{metric}"] = np.quantile(values, .025)
            row[f"q975_{metric}"] = np.quantile(values, .975)
        output.append(row)
    return output


def ensemble_summary(seed_rows, ensemble_rows) -> list[dict]:
    output = []
    keys = sorted({(int(r["fold"]), r["scenario"], r["method"]) for r in seed_rows})
    ensemble = {(int(r["fold"]), r["scenario"], r["method"]): r for r in ensemble_rows}
    for key in keys:
        subset = [r for r in seed_rows if
                  (int(r["fold"]), r["scenario"], r["method"]) == key]
        e = ensemble.get(key, {})
        values = np.asarray([float(r["mae"]) for r in subset])
        output.append({"fold": key[0], "scenario": key[1], "method": key[2],
                       "n_seeds": len(values), "mean_individual_seed_mae": values.mean(),
                       "sd_individual_seed_mae": values.std(ddof=1) if len(values) > 1 else 0,
                       "ensemble_mae": e.get("mae", "")})
    return output


def main() -> None:
    cli = parser().parse_args()
    predictions = read(cli.predictions)
    draws = reliability_rows(predictions)
    write(cli.output_dir / "split_seed_reliability_draws.csv", draws)
    write(cli.output_dir / "split_seed_reliability_summary.csv", reliability_summary(draws))
    summary = ensemble_summary(read(cli.fold_seed_metrics), read(cli.fold_ensemble_metrics))
    write(cli.output_dir / "fold_individual_vs_ensemble_mae.csv", summary)
    print(f"wrote no-retraining robustness analyses to {cli.output_dir}")


if __name__ == "__main__":
    main()
