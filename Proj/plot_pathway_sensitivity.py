#!/usr/bin/env python3
"""Plot the unordered/positional pathway-incidence sensitivity trend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_MPL_CACHE = Path(__file__).resolve().parent / "cache" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.pyplot as plt
import pandas as pd


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path,
                   default=Path("results/pathway_sensitivity/metrics.csv"))
    p.add_argument("--output", type=Path,
                   default=Path("../results/dissertation_figures/pathway_incidence_sensitivity.pdf"))
    return p


def main() -> None:
    cli = parser().parse_args()
    rows = pd.read_csv(cli.metrics)
    summary = (rows.groupby(["cutoff", "sigma", "variant"])["test_mae"]
               .agg(["mean", "std"]).reset_index())
    settings = (summary[["cutoff", "sigma"]].drop_duplicates()
                .sort_values(["cutoff", "sigma"]).reset_index(drop=True))
    labels = [rf"$({r.cutoff:g},{r.sigma:g})$" for r in settings.itertuples()]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    styles = {
        "unordered": ("Unordered trajectory", "#777777", "o"),
        "positional": ("Positional trajectory", "#1f77b4", "D"),
    }
    for variant, (label, colour, marker) in styles.items():
        part = summary[summary.variant == variant].sort_values(["cutoff", "sigma"])
        ax.errorbar(range(len(part)), part["mean"], yerr=part["std"],
                    label=label, color=colour, marker=marker, linewidth=2,
                    markersize=6, capsize=4)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel(r"Pathway-incidence setting $(r_c,\sigma_d)$ (Å)")
    ax.set_ylabel("Test MAE (eV)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cli.output, bbox_inches="tight")
    fig.savefig(cli.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"wrote {cli.output} and {cli.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
