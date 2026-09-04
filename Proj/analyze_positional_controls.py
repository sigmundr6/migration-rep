#!/usr/bin/env python3
"""Summarise completed positional-control predictions and paired contrasts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=Path("results/positional_controls/per_sample_predictions.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/positional_controls"))
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted({int(r["seed"]) for r in rows})
    variants = sorted({r["variant"] for r in rows})
    lookup = {(r["variant"], r["orientation"], int(r["seed"]), r["source"]):
              float(r["absolute_error"]) for r in rows}
    summaries = []
    for variant in variants:
        for orientation in ("forward", "inference_symmetrised", "reverse"):
            seed_maes = []
            for seed in seeds:
                values = [float(r["absolute_error"]) for r in rows
                          if r["variant"] == variant and r["orientation"] == orientation
                          and int(r["seed"]) == seed]
                if values:
                    seed_maes.append(float(np.mean(values)))
            if seed_maes:
                summaries.append({"variant": variant, "orientation": orientation,
                                  "n_seeds": len(seed_maes), "mae_mean": np.mean(seed_maes),
                                  "mae_sd": np.std(seed_maes, ddof=1)})
    write(args.output_dir / "summary.csv", summaries)

    comparisons = (
        ("unordered_minus_shuffled_s", "unordered", "forward", "shuffled_s", "forward"),
        ("shuffled_s_minus_positional", "shuffled_s", "forward", "positional", "forward"),
        ("unordered_minus_positional", "unordered", "forward", "positional", "forward"),
        ("unordered_minus_invariant", "unordered", "forward", "positional_invariant", "forward"),
        ("positional_minus_invariant", "positional", "forward", "positional_invariant", "forward"),
        ("positional_forward_minus_inference_sym", "positional", "forward",
         "positional", "inference_symmetrised"),
    )
    rng = np.random.default_rng(args.seed)
    contrasts = []
    for name, lv, lo, rv, ro in comparisons:
        sources = sorted({key[3] for key in lookup if key[:2] == (lv, lo)})
        differences = []
        for source in sources:
            values = [lookup[(lv, lo, seed, source)] - lookup[(rv, ro, seed, source)]
                      for seed in seeds
                      if (lv, lo, seed, source) in lookup and (rv, ro, seed, source) in lookup]
            if values:
                differences.append(float(np.mean(values)))
        differences = np.asarray(differences)
        draws = rng.choice(differences, size=(args.bootstrap_samples, len(differences)),
                           replace=True).mean(axis=1)
        contrasts.append({"comparison": name, "n_hops": len(differences),
                          "mean_paired_improvement": differences.mean(),
                          "median_paired_improvement": np.median(differences),
                          "ci95_low": np.quantile(draws, .025),
                          "ci95_high": np.quantile(draws, .975),
                          "bootstrap_probability_gt_zero": np.mean(draws > 0)})
    write(args.output_dir / "paired_bootstrap.csv", contrasts)
    print(f"wrote positional-control summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
