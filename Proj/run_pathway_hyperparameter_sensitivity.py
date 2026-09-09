#!/usr/bin/env python3
"""Small, frozen-grid sensitivity check for pathway cutoff and Gaussian width."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from litraj_hypergraphs.data import materialize_prepared, prepare_litraj
from litraj_hypergraphs.experiment import move_splits, release_device_cache, resolve_device, train
from run_positional_controls import make_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    p.add_argument("--cutoffs", type=float, nargs="+", default=[2.5, 3.0, 3.5])
    p.add_argument("--sigmas", type=float, nargs="+", default=[1.2, 1.5, 1.8])
    p.add_argument("--full-grid", action="store_true",
                   help="evaluate every cutoff/sigma combination; by default, pair values positionally")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--neighbour-cutoff", type=float, default=3.5)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/pathway_sensitivity"))
    p.add_argument("--output", type=Path, default=Path("results/pathway_sensitivity/metrics.csv"))
    p.add_argument("--preprocessed-cache-dir", type=Path, default=Path("cache/litraj_preprocessed"))
    p.add_argument("--rebuild-preprocessed-cache", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def main() -> None:
    cli = parser().parse_args()
    # make_args expects a variant list only when called by its own main routine.
    cli.variants = ["unordered", "positional"]
    # Use the central setting to build the shared prepared-data template; each
    # materialized run below then overrides these two incidence parameters.
    cli.hyperedge_cutoff = cli.cutoffs[len(cli.cutoffs) // 2]
    cli.hyperedge_sigma = cli.sigmas[len(cli.sigmas) // 2]
    device = resolve_device(cli.device)
    template = make_args(cli, cli.seeds[0], "unordered")
    prepared = prepare_litraj(cli.data_root, "nebDFT2k", template)
    rows = []
    if cli.full_grid:
        settings = [(cutoff, sigma) for cutoff in cli.cutoffs for sigma in cli.sigmas]
    else:
        if len(cli.cutoffs) != len(cli.sigmas):
            raise ValueError("paired sensitivity requires equally many --cutoffs and --sigmas")
        settings = list(zip(cli.cutoffs, cli.sigmas, strict=True))
    for cutoff, sigma in settings:
        for variant, candidate in (("unordered", "trajectory"), ("positional", "positional")):
            for seed in cli.seeds:
                args = make_args(cli, seed, variant)
                args.hyperedge_cutoff = cutoff
                args.hyperedge_sigma = sigma
                args.checkpoint_dir = cli.checkpoint_dir / f"rc{cutoff:g}_sigma{sigma:g}" / variant
                splits = materialize_prepared(prepared, candidate, args)
                storage = torch.device("cpu") if device.type == "mps" else device
                splits = move_splits(splits, storage)
                model, metrics = train(
                    splits, args, device,
                    f"sensitivity_rc{cutoff:g}_sigma{sigma:g}_{variant}",
                )
                rows.append({"cutoff": cutoff, "sigma": sigma, "variant": variant,
                             "seed": seed, "test_mae": metrics["test"]})
                cli.output.parent.mkdir(parents=True, exist_ok=True)
                with cli.output.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                del model, splits
                release_device_cache(device, f"sensitivity_{variant}_{seed}")
    print(f"wrote {len(rows)} runs to {cli.output}")

    grouped = {}
    for row in rows:
        grouped.setdefault((row["cutoff"], row["sigma"]), {})\
            .setdefault(row["variant"], []).append(row["test_mae"])
    summary_rows = []
    for (cutoff, sigma), variants in grouped.items():
        unordered = np.asarray(variants["unordered"], dtype=float)
        positional = np.asarray(variants["positional"], dtype=float)
        summary_rows.append({
            "cutoff": cutoff, "sigma": sigma, "n_seeds": len(unordered),
            "unordered_mean": unordered.mean(), "unordered_sd": unordered.std(ddof=1),
            "positional_mean": positional.mean(), "positional_sd": positional.std(ddof=1),
            "paired_improvement_mean": (unordered - positional).mean(),
            "paired_improvement_sd": (unordered - positional).std(ddof=1),
            "seeds_favouring_positional": int(np.sum(positional < unordered)),
        })
    summary_path = cli.output.with_name("summary.csv")
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    print(f"wrote sensitivity summary to {summary_path}")


if __name__ == "__main__":
    main()
