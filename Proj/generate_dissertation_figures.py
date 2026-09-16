#!/usr/bin/env python3
"""Generate publication-ready dissertation figures from completed experiments.

The script is read-only with respect to experiment outputs.  It creates PNG and
PDF versions of each figure so the raster files can be previewed while the PDF
files can be included in LaTeX without loss of resolution.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exploratory-dir",
        type=Path,
        default=ROOT / "results" / "exploratory_statistics",
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=ROOT / "results" / "final_statistics",
    )
    parser.add_argument(
        "--grouped-dir",
        type=Path,
        default=ROOT / "Proj" / "results" / "material_disjoint_evaluation",
    )
    parser.add_argument(
        "--cross-fidelity-csv",
        type=Path,
        default=ROOT / "results" / "nebdft2k_cross_fidelity.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "dissertation_figures",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def configure_matplotlib(output: Path):
    cache = output / ".matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )
    return plt


def save(fig, output: Path, stem: str, dpi: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=dpi, facecolor="white")
    fig.savefig(output / f"{stem}.pdf", facecolor="white")


def representation_hierarchy(plt, output: Path, dpi: int) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.0, 3.15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.15)
    ax.axis("off")
    labels = [
        ("Crystal", "global structure"),
        ("Midpoint", "+ event location"),
        ("Trajectory", "+ pathway membership"),
        ("Positional", "+ reaction-coordinate\nposition"),
    ]
    colors = ["#d9d9d9", "#bdd7e7", "#6baed6", "#2171b5"]
    xs = [0.2, 2.7, 5.2, 7.7]
    for index, ((title, detail), color, x) in enumerate(zip(labels, colors, xs)):
        box = FancyBboxPatch(
            (x, 1.05), 2.05, 1.25,
            boxstyle="round,pad=0.04,rounding_size=0.05",
            linewidth=1.2, edgecolor="#333333", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + 1.025, 1.82, title, ha="center", va="center", weight="bold",
                color="white" if index == 3 else "#111111", fontsize=11)
        ax.text(x + 1.025, 1.41, detail, ha="center", va="center",
                color="white" if index == 3 else "#222222", fontsize=8.2,
                linespacing=1.15)
        if index < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x + 2.08, 1.68), (xs[index + 1] - 0.08, 1.68),
                                         arrowstyle="-|>", mutation_scale=13,
                                         linewidth=1.2, color="#444444"))
    ax.text(5, 2.82, "Controlled hierarchy of migration-event information",
            ha="center", va="center", fontsize=13, weight="bold")
    ax.text(5, 0.48,
            "Each model uses the same periodic crystal encoder; only the event representation changes.",
            ha="center", va="center", fontsize=9)
    save(fig, output, "representation_hierarchy", dpi)
    plt.close(fig)


def main_mae_comparison(plt, exploratory: Path, output: Path, dpi: int) -> None:
    rows = read_csv(exploratory / "summary.csv")
    lookup = {row["method"]: row for row in rows}
    core = ["crystal", "midpoint", "trajectory", "positional"]
    alternatives = [
        "randomized", "segmented", "geometric_bottleneck", "coordination_change",
        "species_image", "image_coordination", "position_aware_trajectory_graph",
        "position_aware_trajectory_graph_shuffled",
    ]
    order = core + alternatives
    names = {
        "crystal": "Crystal",
        "randomized": "Randomised pathway",
        "midpoint": "Midpoint",
        "trajectory": "Unordered trajectory",
        "segmented": "Segmented trajectory",
        "geometric_bottleneck": "Geometric bottleneck",
        "coordination_change": "Coordination change",
        "species_image": "Species–image",
        "image_coordination": "Image coordination",
        "position_aware_trajectory_graph": "Position-aware graph",
        "position_aware_trajectory_graph_shuffled": "Shuffled graph adjacency",
        "positional": "Positional trajectory",
    }
    missing = [method for method in order if method not in lookup]
    if missing:
        raise ValueError(f"Missing methods in exploratory summary: {missing}")
    upper = max(float(lookup[m]["mae_mean"]) + float(lookup[m]["mae_std"]) for m in order) + .07
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.1), sharex=True)
    for ax, methods, title, color in (
        (axes[0], core, "A  Controlled information hierarchy", "#2171b5"),
        (axes[1], alternatives, "B  Supporting mechanistic alternatives", "#969696"),
    ):
        means = np.array([float(lookup[m]["mae_mean"]) for m in methods])
        stds = np.array([float(lookup[m]["mae_std"]) for m in methods])
        y = np.arange(len(methods))
        ax.errorbar(means, y, xerr=stds, fmt="none", ecolor="#555555", capsize=3,
                    linewidth=1.1, zorder=1)
        ax.scatter(means, y, color=color, s=48, edgecolor="white", linewidth=.6, zorder=2)
        for x, yi in zip(means, y):
            ax.text(x + .012, yi, f"{x:.3f}", va="center", fontsize=8)
        ax.set_yticks(y, [names[m] for m in methods])
        ax.invert_yaxis(); ax.set_title(title); ax.grid(axis="x", alpha=.22)
        ax.set_xlim(.25, upper)
    fig.supxlabel("Test MAE (eV; mean ± SD across 10 seeds)")
    fig.suptitle("Information-matched core and heterogeneous-input alternatives", weight="bold")
    fig.tight_layout()
    save(fig, output, "representation_mae_comparison", dpi)
    plt.close(fig)


def cross_fidelity_ladder(plt, final_dir: Path, cross_csv: Path,
                          output: Path, dpi: int) -> None:
    summary = read_csv(final_dir / "summary.csv")
    lookup = {(r["scenario"], r["method"]): float(r["mae_mean"]) for r in summary}
    cross = read_csv(cross_csv)
    raw_candidates = [r for r in cross if r.get("method") == "raw_em_bvse"]
    if not raw_candidates:
        raise ValueError("raw_em_bvse row is missing from cross-fidelity CSV")
    values = [
        float(raw_candidates[0]["test"]),
        lookup[("affine_bvse", "affine_em_bvse")],
        lookup[("bvse_scalar_path_fusion", "positional")],
        lookup[("bvse_calibrated_delta", "positional")],
    ]
    labels = ["Raw BVSE", "Affine calibration", "Scalar + positional path",
              "Calibrated residual model"]
    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    ax.plot(x, values, color="#737373", linewidth=1.7, zorder=1)
    ax.scatter(x, values, c=["#bdbdbd", "#6baed6", "#3182bd", "#08519c"],
               s=85, edgecolor="white", linewidth=0.8, zorder=2)
    for xi, value in zip(x, values):
        ax.text(xi, value + 0.035, f"{value:.3f} eV", ha="center", weight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("Progressive use of low-cost BVSE information")
    ax.set_ylim(0.20, max(values) + 0.11)
    ax.grid(axis="y", alpha=0.22)
    save(fig, output, "cross_fidelity_performance_ladder", dpi)
    plt.close(fig)


def ensemble_predictions(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"target": [], "prediction": [], "absolute_error": []}
    )
    for row in rows:
        key = (row["scenario"], row["method"], row["source"])
        grouped[key]["target"].append(float(row["target"]))
        grouped[key]["prediction"].append(float(row["prediction"]))
        grouped[key]["absolute_error"].append(float(row["absolute_error"]))
    output = {}
    for key, fields in grouped.items():
        target = float(np.mean(fields["target"]))
        prediction = float(np.mean(fields["prediction"]))
        output[key] = {
            "target": target,
            "prediction": prediction,
            # Per-hop comparisons in the report use the error of the
            # seed-ensemble prediction, not the mean error of individual seeds.
            "absolute_error": abs(target - prediction),
        }
    return output


def paired_improvement_distributions(plt, final_dir: Path, output: Path, dpi: int) -> None:
    ensemble = ensemble_predictions(read_csv(final_dir / "per_sample_predictions.csv"))
    keys = defaultdict(dict)
    for (scenario, method, source), values in ensemble.items():
        keys[(scenario, method)][source] = values
    comparisons = [
        (("dft_path", "midpoint"), ("dft_path", "positional"),
         "Positional − midpoint"),
        (("affine_bvse", "affine_em_bvse"), ("bvse_calibrated_delta", "positional"),
         "Delta − affine BVSE"),
    ]
    deltas = []
    for left, right, _ in comparisons:
        sources = sorted(set(keys[left]) & set(keys[right]))
        if not sources:
            raise ValueError(f"No matched sources for {left} versus {right}")
        deltas.append(np.array([keys[left][s]["absolute_error"] -
                                keys[right][s]["absolute_error"] for s in sources]))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), sharey=True)
    rng = np.random.default_rng(2026)
    for ax, delta, (_, _, label) in zip(axes, deltas, comparisons):
        violin = ax.violinplot(delta, positions=[0], widths=0.75, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor("#6baed6")
            body.set_edgecolor("#2171b5")
            body.set_alpha(0.55)
        jitter = rng.uniform(-0.16, 0.16, len(delta))
        ax.scatter(jitter, delta, s=9, alpha=0.24, color="#08519c", linewidths=0)
        ax.scatter([0], [np.mean(delta)], marker="D", s=38, color="#cb181d", zorder=4,
                   label="Mean")
        ax.axhline(0, color="#333333", linewidth=1)
        helped = 100 * np.mean(delta > 0)
        ax.set_title(f"{label}\n{helped:.1f}% of hops improved")
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Paired reduction in absolute error (eV)\npositive values favour pathway-aware model")
    save(fig, output, "paired_per_hop_improvements", dpi)
    plt.close(fig)


def grouped_fold_consistency(plt, grouped_dir: Path, output: Path, dpi: int) -> None:
    rows = read_csv(grouped_dir / "paired_fold_effects.csv")
    groups = defaultdict(list)
    for row in rows:
        groups[row["comparison"]].append((int(row["fold"]), float(row["mae_improvement"])))
    order = ["crystal_minus_midpoint", "midpoint_minus_trajectory",
             "midpoint_minus_positional", "trajectory_minus_positional",
             "affine_minus_delta"]
    labels = {
        "crystal_minus_midpoint": "Crystal → midpoint",
        "midpoint_minus_trajectory": "Midpoint → unordered trajectory",
        "midpoint_minus_positional": "Midpoint → positional",
        "trajectory_minus_positional": "Unordered trajectory → positional",
        "affine_minus_delta": "Affine BVSE → positional delta",
    }
    missing = [name for name in order if name not in groups]
    if missing:
        raise ValueError(f"Missing grouped comparisons: {missing}; available={sorted(groups)}")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for yi, name in enumerate(order):
        points = sorted(groups[name])
        values = np.array([v for _, v in points])
        ax.scatter(values, np.full(len(values), yi), color="#9ecae1", s=34,
                   edgecolor="#2171b5", linewidth=0.7, zorder=2)
        ax.scatter([values.mean()], [yi], marker="D", color="#cb181d", s=42, zorder=3)
        ax.plot([values.min(), values.max()], [yi, yi], color="#737373", linewidth=1, zorder=1)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(range(len(order)), [labels[name] for name in order])
    ax.invert_yaxis()
    ax.set_xlabel("Fold MAE improvement (eV; positive favours model on right)")
    ax.set_title("Parent-material-disjoint effect consistency")
    ax.grid(axis="x", alpha=0.2)
    save(fig, output, "material_disjoint_fold_effects", dpi)
    plt.close(fig)


def parity_panels(plt, final_dir: Path, output: Path, dpi: int) -> None:
    ensemble = ensemble_predictions(read_csv(final_dir / "per_sample_predictions.csv"))
    panels = [
        ("dft_path", "crystal", "Crystal"),
        ("dft_path", "midpoint", "Midpoint"),
        ("dft_path", "positional", "DFT positional path"),
        ("affine_bvse", "affine_em_bvse", "Affine BVSE"),
        ("bvse_calibrated_delta", "positional", "BVSE positional delta"),
    ]
    data = []
    for scenario, method, label in panels:
        points = [v for (s, m, _), v in ensemble.items() if s == scenario and m == method]
        if not points:
            raise ValueError(f"No predictions for {scenario}/{method}")
        data.append((label, np.array([p["target"] for p in points]),
                     np.array([p["prediction"] for p in points])))
    maximum = max(float(max(np.max(y), np.max(p))) for _, y, p in data)
    limit = np.ceil(maximum * 2) / 2
    fig, axes = plt.subplots(2, 3, figsize=(9.3, 6.1), sharex=True, sharey=True)
    for ax, (label, target, prediction) in zip(axes.flat, data):
        ax.scatter(target, prediction, s=12, alpha=0.48, color="#2171b5", linewidths=0)
        ax.plot([0, limit], [0, limit], linestyle="--", color="#555555", linewidth=1)
        mae = np.mean(np.abs(target - prediction))
        ax.set_title(f"{label}\nMAE = {mae:.3f} eV")
        ax.grid(alpha=0.15)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
    axes.flat[-1].axis("off")
    fig.supxlabel("DFT migration barrier (eV)")
    fig.supylabel("Predicted migration barrier (eV)")
    fig.suptitle("Seed-ensemble prediction parity", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, output, "prediction_parity_panels", dpi)
    plt.close(fig)


def core_validation_curves(plt, exploratory: Path, output: Path, dpi: int) -> None:
    rows = read_csv(exploratory / "validation_curves.csv")
    methods = ["crystal", "midpoint", "trajectory", "positional"]
    labels = {"crystal": "Crystal", "midpoint": "Midpoint",
              "trajectory": "Unordered trajectory", "positional": "Positional"}
    colors = {"crystal": "#969696", "midpoint": "#6baed6",
              "trajectory": "#31a354", "positional": "#08519c"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        by_epoch = defaultdict(list)
        for row in subset:
            by_epoch[int(row["epoch"])].append(float(row["val_mae"]))
        epochs = np.array(sorted(by_epoch))
        means = np.array([np.mean(by_epoch[e]) for e in epochs])
        sem = np.array([np.std(by_epoch[e], ddof=1) / np.sqrt(len(by_epoch[e]))
                        if len(by_epoch[e]) > 1 else 0.0 for e in epochs])
        ax.plot(epochs, means, label=labels[method], color=colors[method], linewidth=1.8)
        ax.fill_between(epochs, means - sem, means + sem, color=colors[method], alpha=0.14)
    ax.axvline(25, linestyle="--", color="#555555", linewidth=1,
               label="Previous 25-epoch horizon")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (eV; mean ± SEM)")
    ax.set_title("Validation convergence of core representations")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    save(fig, output, "core_validation_curves", dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plt = configure_matplotlib(args.output_dir)
    representation_hierarchy(plt, args.output_dir, args.dpi)
    main_mae_comparison(plt, args.exploratory_dir, args.output_dir, args.dpi)
    cross_fidelity_ladder(plt, args.final_dir, args.cross_fidelity_csv,
                          args.output_dir, args.dpi)
    paired_improvement_distributions(plt, args.final_dir, args.output_dir, args.dpi)
    grouped_fold_consistency(plt, args.grouped_dir, args.output_dir, args.dpi)
    parity_panels(plt, args.final_dir, args.output_dir, args.dpi)
    core_validation_curves(plt, args.exploratory_dir, args.output_dir, args.dpi)
    print(f"Wrote dissertation figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
