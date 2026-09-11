#!/usr/bin/env python3
"""Explain which LiTraj test hops benefit from pathway-aware representations.

The analysis is intentionally hypothesis-driven.  It compares seed-averaged
predictions and relates the resulting per-hop error improvement to barrier
height, BVSE disagreement, path geometry, and local coordination/bottlenecks.
No model training is performed.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from litraj_hypergraphs.data import PreparedTrajectory, prepare_litraj
from litraj_hypergraphs.geometry import periodic_distances, unwrap_points


HYPOTHESES = {
    "dft_barrier": "DFT migration barrier",
    "affine_bvse_absolute_error": "absolute calibrated-BVSE error",
    "dft_path_length": "DFT path length",
    "dft_tortuosity": "DFT path tortuosity",
    "dft_bottleneck_mean_k_distance": "DFT bottleneck clearance",
    "dft_coordination_range": "DFT coordination variation",
    "bvse_path_length": "BVSE path length",
    "bvse_tortuosity": "BVSE path tortuosity",
    "bvse_bottleneck_mean_k_distance": "BVSE bottleneck clearance",
    "bvse_coordination_range": "BVSE coordination variation",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--q3-predictions", type=Path,
                   default=Path("results/final_statistics/per_sample_predictions.csv"),
                   help="per-hop file containing the final dft_path midpoint and positional runs")
    p.add_argument("--final-predictions", type=Path,
                   default=Path("results/final_statistics/per_sample_predictions.csv"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--preprocessed-cache-dir", type=Path,
                   default=Path("cache/litraj_preprocessed"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/per_hop_pathway_analysis"))
    p.add_argument("--neighbour-cutoff", type=float, default=3.5,
                   help="must match the cutoff used to create the preprocessing cache")
    p.add_argument("--coordination-cutoff", type=float, default=3.0)
    p.add_argument("--coordination-sigma", type=float, default=1.5)
    p.add_argument("--bottleneck-neighbours", type=int, default=4)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    p.add_argument("--quantile-bins", type=int, default=4)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--min-family-size", type=int, default=8)
    return p


def mean_predictions(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"source", "target", "prediction", "seed"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    target_spread = frame.groupby("source")["target"].agg(lambda x: x.max() - x.min())
    if (target_spread > 1e-6).any():
        raise ValueError(f"{name}: targets differ between seeds for the same hop")
    out = frame.groupby("source", as_index=False).agg(
        target=("target", "mean"), prediction=("prediction", "mean"),
        prediction_sd=("prediction", "std"), seeds=("seed", "nunique"),
    )
    return out.rename(columns={
        "prediction": f"{name}_prediction", "prediction_sd": f"{name}_prediction_sd",
        "seeds": f"{name}_seeds",
    })


def select(df: pd.DataFrame, **values) -> pd.DataFrame:
    result = df
    for column, value in values.items():
        if column not in result:
            raise ValueError(f"prediction file has no {column!r} column")
        result = result[result[column] == value]
    if result.empty:
        raise ValueError(f"no predictions match {values}")
    return result


def build_improvements(q3: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    if "scenario" in q3:
        q3 = select(q3, scenario="dft_path")
    midpoint = mean_predictions(select(q3, method="midpoint"), "midpoint")
    positional = mean_predictions(select(q3, method="positional"), "positional")
    affine = mean_predictions(select(final, scenario="affine_bvse", method="affine_em_bvse"),
                              "affine_bvse")
    delta = mean_predictions(select(final, scenario="bvse_calibrated_delta", method="positional"),
                             "bvse_delta")
    merged = midpoint.merge(positional, on="source", suffixes=("_mid", "_pos"), validate="one_to_one")
    merged = merged.merge(affine, on="source", validate="one_to_one")
    merged = merged.merge(delta, on="source", validate="one_to_one")
    target_columns = [c for c in merged if c.startswith("target")]
    targets = merged[target_columns].to_numpy(float)
    if np.max(np.ptp(targets, axis=1)) > 1e-5:
        raise ValueError("Q3 and Q4 prediction targets do not agree")
    merged["dft_barrier"] = targets.mean(axis=1)
    merged["midpoint_absolute_error"] = abs(merged.dft_barrier - merged.midpoint_prediction)
    merged["positional_absolute_error"] = abs(merged.dft_barrier - merged.positional_prediction)
    merged["delta_pos"] = merged.midpoint_absolute_error - merged.positional_absolute_error
    merged["affine_bvse_absolute_error"] = abs(
        merged.dft_barrier - merged.affine_bvse_prediction)
    merged["bvse_delta_absolute_error"] = abs(merged.dft_barrier - merged.bvse_delta_prediction)
    merged["delta_bvse"] = (
        merged.affine_bvse_absolute_error - merged.bvse_delta_absolute_error)
    return merged.drop(columns=target_columns)


def composition(numbers: np.ndarray) -> tuple[str, str]:
    try:
        from ase.data import chemical_symbols
        symbol = lambda z: chemical_symbols[int(z)]
    except ImportError:
        symbol = lambda z: f"Z{int(z)}"
    unique, counts = np.unique(numbers, return_counts=True)
    divisor = math.gcd(*map(int, counts))
    formula = "".join(f"{symbol(z)}{'' if c // divisor == 1 else c // divisor}"
                      for z, c in zip(unique, counts))
    anions = [symbol(z) for z in unique if int(z) in {7, 8, 9, 16, 17, 34, 35, 52, 53}]
    return formula, "+".join(anions) if anions else "other"


def trajectory_descriptors(item: PreparedTrajectory, prefix: str, args) -> dict[str, object]:
    li_path = unwrap_points(item.frames[:, item.migrating_index], item.cell)
    steps = np.linalg.norm(np.diff(li_path, axis=0), axis=1)
    length = float(steps.sum())
    displacement = float(np.linalg.norm(li_path[-1] - li_path[0]))
    tortuosity = length / displacement if displacement > 1e-8 else np.nan
    mask = np.arange(len(item.numbers)) != item.migrating_index
    hard_coordination, smooth_coordination, nearest_k = [], [], []
    for frame in item.frames:
        distances = periodic_distances(frame[mask], frame[item.migrating_index], item.cell)
        ordered = np.sort(distances)
        k = min(args.bottleneck_neighbours, len(ordered))
        nearest_k.append(float(ordered[:k].mean()) if k else np.nan)
        nearby = distances[distances <= args.coordination_cutoff]
        hard_coordination.append(float(len(nearby)))
        smooth_coordination.append(float(np.exp(-(nearby / args.coordination_sigma) ** 2).sum()))
    bottleneck_index = int(np.nanargmin(nearest_k))
    path_coordinate = np.concatenate([[0.0], np.cumsum(steps)])
    path_coordinate = path_coordinate / length if length > 1e-8 else np.zeros(len(li_path))
    return {
        f"{prefix}_path_length": length,
        f"{prefix}_endpoint_displacement": displacement,
        f"{prefix}_tortuosity": tortuosity,
        f"{prefix}_bottleneck_mean_k_distance": nearest_k[bottleneck_index],
        f"{prefix}_bottleneck_path_coordinate": float(path_coordinate[bottleneck_index]),
        f"{prefix}_coordination_at_bottleneck": hard_coordination[bottleneck_index],
        f"{prefix}_smooth_coordination_at_bottleneck": smooth_coordination[bottleneck_index],
        f"{prefix}_coordination_range": float(np.ptp(hard_coordination)),
        f"{prefix}_smooth_coordination_range": float(np.ptp(smooth_coordination)),
    }


def load_descriptors(args) -> pd.DataFrame:
    cache_name = (f"trajectory_descriptors_coord{args.coordination_cutoff:g}_"
                  f"sigma{args.coordination_sigma:g}_k{args.bottleneck_neighbours}.csv")
    descriptor_cache = args.output_dir / cache_name
    if descriptor_cache.exists():
        print(f"Loaded trajectory descriptors from {descriptor_cache}", flush=True)
        return pd.read_csv(descriptor_cache)
    shared = SimpleNamespace(
        neighbour_cutoff=args.neighbour_cutoff,
        target_source="em_dft", preprocessed_cache_dir=args.preprocessed_cache_dir,
        rebuild_preprocessed_cache=False,
    )
    by_source: dict[str, dict[str, object]] = {}
    for trajectory_source, prefix in (("relaxed", "dft"), ("init", "bvse")):
        shared.trajectory_source = trajectory_source
        prepared = prepare_litraj(args.data_root, "nebDFT2k", shared)
        for item in prepared["test"]:
            row = by_source.setdefault(item.source, {"source": item.source})
            row.update(trajectory_descriptors(item, prefix, args))
            if prefix == "dft":
                row["composition_family"], row["anion_family"] = composition(item.numbers)
                row["em_bvse"] = item.cheap_barrier
    descriptors = pd.DataFrame(by_source.values())
    descriptors.to_csv(descriptor_cache, index=False)
    print(f"Saved trajectory descriptors to {descriptor_cache}", flush=True)
    return descriptors


def ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(float)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
        return np.nan
    return float(np.corrcoef(ranks(x[valid]), ranks(y[valid]))[0, 1])


def correlation_table(data: pd.DataFrame, args) -> pd.DataFrame:
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    variables = [v for v in HYPOTHESES if v in data]
    for outcome in ("delta_pos", "delta_bvse"):
        for variable in variables:
            subset = data[[variable, outcome]].dropna().to_numpy(float)
            observed = spearman(subset[:, 0], subset[:, 1])
            boot = []
            for _ in range(args.bootstrap_samples):
                sampled = subset[rng.integers(0, len(subset), len(subset))]
                boot.append(spearman(sampled[:, 0], sampled[:, 1]))
            boot = np.asarray(boot)
            rows.append({
                "outcome": outcome, "variable": variable,
                "hypothesis": HYPOTHESES[variable], "n": len(subset),
                "spearman_rho": observed,
                "bootstrap_ci_low": np.nanquantile(boot, 0.025),
                "bootstrap_ci_high": np.nanquantile(boot, 0.975),
            })
    return pd.DataFrame(rows)


def binned_table(data: pd.DataFrame, args) -> pd.DataFrame:
    rng = np.random.default_rng(args.bootstrap_seed + 1)
    rows = []
    for outcome in ("delta_pos", "delta_bvse"):
        for variable in [v for v in HYPOTHESES if v in data]:
            subset = data[[variable, outcome]].dropna().copy()
            try:
                subset["bin"] = pd.qcut(subset[variable], args.quantile_bins, duplicates="drop")
            except ValueError:
                continue
            for interval, group in subset.groupby("bin", observed=True):
                values = group[outcome].to_numpy(float)
                means = np.asarray([rng.choice(values, len(values), replace=True).mean()
                                    for _ in range(args.bootstrap_samples)])
                rows.append({
                    "outcome": outcome, "variable": variable, "bin": str(interval),
                    "n": len(values), "variable_mean": group[variable].mean(),
                    "mean_improvement": values.mean(), "median_improvement": np.median(values),
                    "fraction_helped": np.mean(values > 0),
                    "mean_ci_low": np.quantile(means, 0.025),
                    "mean_ci_high": np.quantile(means, 0.975),
                })
    return pd.DataFrame(rows)


def ranked_examples(data: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows = []
    for outcome in ("delta_pos", "delta_bvse"):
        ordered = data.sort_values(outcome)
        for label, group in (("largest_harm", ordered.head(top_k)),
                             ("largest_help", ordered.tail(top_k).iloc[::-1])):
            copy = group.copy()
            copy.insert(0, "effect", label)
            copy.insert(0, "outcome", outcome)
            rows.append(copy)
    return pd.concat(rows, ignore_index=True)


def family_table(data: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for family_column in ("composition_family", "anion_family"):
        for family, group in data.groupby(family_column):
            if len(group) < args.min_family_size:
                continue
            for outcome in ("delta_pos", "delta_bvse"):
                values = group[outcome].to_numpy(float)
                rows.append({"family_type": family_column, "family": family, "n": len(group),
                             "outcome": outcome, "mean_improvement": values.mean(),
                             "median_improvement": np.median(values),
                             "fraction_helped": np.mean(values > 0)})
    return pd.DataFrame(rows)


def make_plots(data: pd.DataFrame, correlations: pd.DataFrame, output: Path) -> None:
    cache = output / ".matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    import matplotlib.pyplot as plt

    choices = [
        ("dft_barrier", "delta_pos"), ("dft_tortuosity", "delta_pos"),
        ("affine_bvse_absolute_error", "delta_bvse"),
        ("bvse_bottleneck_mean_k_distance", "delta_bvse"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, (x, y) in zip(axes.flat, choices):
        ax.axhline(0, color="0.5", lw=1)
        ax.scatter(data[x], data[y], s=18, alpha=.55)
        row = correlations[(correlations.variable == x) & (correlations.outcome == y)].iloc[0]
        ax.set(xlabel=HYPOTHESES[x], ylabel=f"{y} (eV)",
               title=f"Spearman rho={row.spearman_rho:.2f} "
                     f"[{row.bootstrap_ci_low:.2f}, {row.bootstrap_ci_high:.2f}]")
    fig.savefig(output / "hypothesis_scatterplots.png", dpi=200)
    plt.close(fig)


def write_summary(data: pd.DataFrame, correlations: pd.DataFrame, path: Path) -> None:
    lines = ["Per-hop pathway-benefit analysis", "=" * 32, ""]
    for outcome, description in (
        ("delta_pos", "midpoint error minus positional error"),
        ("delta_bvse", "affine-BVSE error minus delta-model error"),
    ):
        values = data[outcome]
        lines += [f"{outcome} ({description})", f"  hops: {len(values)}",
                  f"  mean improvement: {values.mean():.4f} eV",
                  f"  median improvement: {values.median():.4f} eV",
                  f"  fraction helped: {(values > 0).mean():.3f}", ""]
        subset = correlations[correlations.outcome == outcome].copy()
        subset["magnitude"] = subset.spearman_rho.abs()
        for row in subset.nlargest(3, "magnitude").itertuples():
            lines.append(f"  {row.variable}: rho={row.spearman_rho:.3f} "
                         f"(95% bootstrap CI {row.bootstrap_ci_low:.3f}, "
                         f"{row.bootstrap_ci_high:.3f})")
        lines.append("")
    lines.append("Positive improvement means the pathway-aware method helped that hop.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    q3 = pd.read_csv(args.q3_predictions)
    final = pd.read_csv(args.final_predictions)
    improvements = build_improvements(q3, final)
    descriptors = load_descriptors(args)
    data = improvements.merge(descriptors, on="source", how="left", validate="one_to_one")
    if data.dft_path_length.isna().any():
        missing = data.loc[data.dft_path_length.isna(), "source"].tolist()
        raise ValueError(f"trajectory descriptors missing for {len(missing)} prediction rows")

    correlations = correlation_table(data, args)
    bins = binned_table(data, args)
    examples = ranked_examples(data, args.top_k)
    families = family_table(data, args)
    data.to_csv(args.output_dir / "per_hop_analysis.csv", index=False)
    correlations.to_csv(args.output_dir / "hypothesis_correlations.csv", index=False)
    bins.to_csv(args.output_dir / "binned_hypothesis_summaries.csv", index=False)
    examples.to_csv(args.output_dir / "largest_help_and_harm.csv", index=False)
    families.to_csv(args.output_dir / "chemistry_family_summaries.csv", index=False)
    write_summary(data, correlations, args.output_dir / "summary.txt")
    make_plots(data, correlations, args.output_dir)
    print((args.output_dir / "summary.txt").read_text(), end="")
    print(f"Wrote analysis outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
