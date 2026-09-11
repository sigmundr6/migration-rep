#!/usr/bin/env python3
"""Relate pathway-model gains to LiTraj classical and path-profile descriptors.

Experiment 1 reproduces the five published LiTraj feature definitions through
``ions.featurizers.EdgeFeaturizer``.  Since nebDFT2k stores vacancy-containing
NEB images rather than the pristine edge structure, the endpoint Li site is
reconstructed from the final image; outputs are labelled accordingly.

Experiment 2 generalises Voronoi volume and neighbour chemistry along every
NEB image.  All extraction is resumable.  Confirmatory tests are limited to the
five published variables; trajectory-profile tests are exploratory and report
Benjamini-Hochberg adjusted p-values.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from litraj_hypergraphs.data import PreparedTrajectory, prepare_litraj
from litraj_hypergraphs.geometry import unwrap_points


OFFICIAL_FEATURES = {
    "litraj_min_volume": "min_volume",
    "litraj_edge_length": "length",
    "litraj_max_weighted_mean_oxidation": "max_mean_oxi_state_weighted",
    "litraj_max_volume": "max_volume",
    "litraj_min_mean_covalent_radius": "min_mean_rc",
}


def decorate_oxidation_states(atoms):
    """Use the notebook's decorator, with pymatgen BV analysis as a robust fallback."""
    from ions import Decorator
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return Decorator().decorate(atoms)
    except ValueError:
        from pymatgen.analysis.bond_valence import BVAnalyzer
        from pymatgen.io.ase import AseAtomsAdaptor
        structure = AseAtomsAdaptor.get_structure(atoms)
        decorated = BVAnalyzer().get_oxi_state_decorated_structure(structure)
        result = atoms.copy()
        result.set_array("oxi_states", np.asarray(
            [site.specie.oxi_state for site in decorated], dtype=float))
        return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-hop-analysis", type=Path,
                   default=Path("results/per_hop_pathway_analysis/per_hop_analysis.csv"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--preprocessed-cache-dir", type=Path,
                   default=Path("cache/litraj_preprocessed"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/litraj_classical_profile_analysis"))
    p.add_argument("--sources", nargs="+", choices=["relaxed", "init"],
                   default=["relaxed", "init"])
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    p.add_argument("--max-hops", type=int,
                   help="development aid; omit for all 220 test hops")
    p.add_argument("--rebuild-descriptors", action="store_true")
    p.add_argument("--neighbour-cutoff", type=float, default=3.5,
                   help="must match the existing preprocessing cache")
    p.add_argument("--top-k", type=int, default=15)
    return p


def reconstructed_edge_features(item: PreparedTrajectory) -> tuple[dict[str, float], str]:
    """Run the official featurizer after restoring the vacant endpoint Li."""
    from ase import Atom, Atoms
    from ions.featurizers import EdgeFeaturizer
    from ions.geom import Edge

    atoms = Atoms(numbers=item.numbers, positions=item.frames[0], cell=item.cell, pbc=True)
    endpoint = unwrap_points(item.frames[:, item.migrating_index], item.cell)[-1]
    atoms.append(Atom("Li", position=endpoint))
    edge = Edge(atoms, item.migrating_index, len(atoms) - 1, [0, 0, 0])
    stats = ["min", "max", "mean", "range"]
    # Four of the five highlighted features do not require oxidation states.
    raw = EdgeFeaturizer(edge, oxi=False).featurize(stats=stats)
    features = {
        output: float(raw[source])
        for output, source in OFFICIAL_FEATURES.items()
        if output != "litraj_max_weighted_mean_oxidation"
    }
    try:
        decorated = decorate_oxidation_states(atoms)
        decorated_edge = Edge(decorated, item.migrating_index, len(decorated) - 1, [0, 0, 0])
        oxidation_raw = EdgeFeaturizer(decorated_edge, oxi=True).featurize(stats=stats)
        features["litraj_max_weighted_mean_oxidation"] = float(
            oxidation_raw[OFFICIAL_FEATURES["litraj_max_weighted_mean_oxidation"]])
        oxidation_status = "ok"
    except Exception as exc:
        features["litraj_max_weighted_mean_oxidation"] = np.nan
        oxidation_status = f"missing: {type(exc).__name__}: {exc}"
    return features, oxidation_status


def reconstructed_framework_oxidation(item: PreparedTrajectory) -> np.ndarray:
    """Assign valences on the charge-balanced structure, then remove restored endpoint."""
    from ase import Atom, Atoms
    atoms = Atoms(numbers=item.numbers, positions=item.frames[0], cell=item.cell, pbc=True)
    endpoint = unwrap_points(item.frames[:, item.migrating_index], item.cell)[-1]
    atoms.append(Atom("Li", position=endpoint))
    decorated = decorate_oxidation_states(atoms)
    return np.asarray(decorated.arrays["oxi_states"][:-1], dtype=float)


def profile_at_images(item: PreparedTrajectory) -> tuple[dict[str, float], str]:
    """Compute Voronoi/local-chemistry profiles at the migrating Li site."""
    from ase import Atoms
    from ase.data import covalent_radii
    from ions.geom.utils import VoroSite

    try:
        oxidation = reconstructed_framework_oxidation(item)
        oxidation_status = "ok"
    except Exception as exc:
        oxidation = None
        oxidation_status = f"missing: {type(exc).__name__}: {exc}"
    radii = covalent_radii[np.asarray(item.numbers, dtype=int)]
    profile_names = ["volume", "effective_cn", "mean_rc"]
    if oxidation is not None:
        profile_names.append("weighted_mean_oxi")
    profiles = {name: [] for name in profile_names}
    for positions in item.frames:
        atoms = Atoms(numbers=item.numbers, positions=positions, cell=item.cell, pbc=True)
        data, _ = VoroSite(atoms).get_poly_data(item.migrating_index)
        neighbours = np.asarray(data["nn_id_unitcell"], dtype=int)
        areas = np.asarray(data["area"], dtype=float)
        weights = np.asarray(data["solid_angle"], dtype=float)
        weights /= weights.max()
        profiles["volume"].append(float(np.asarray(data["volume"]).sum()))
        profiles["effective_cn"].append(float(areas.sum() ** 2 / np.square(areas).sum()))
        profiles["mean_rc"].append(float(radii[neighbours].mean()))
        if oxidation is not None:
            profiles["weighted_mean_oxi"].append(
                float(np.average(oxidation[neighbours], weights=weights)))

    path = unwrap_points(item.frames[:, item.migrating_index], item.cell)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    coordinate = np.r_[0.0, np.cumsum(steps)]
    coordinate = coordinate / coordinate[-1] if coordinate[-1] > 1e-12 else np.linspace(0, 1, len(path))
    result: dict[str, float] = {}
    for name, raw_values in profiles.items():
        values = np.asarray(raw_values, dtype=float)
        minimum, maximum = int(values.argmin()), int(values.argmax())
        midpoint_value = float(np.interp(.5, coordinate, values))
        left_x = np.r_[coordinate[coordinate < .5], .5]
        left_y = np.r_[values[coordinate < .5], midpoint_value]
        right_x = np.r_[.5, coordinate[coordinate > .5]]
        right_y = np.r_[midpoint_value, values[coordinate > .5]]
        left = np.trapezoid(left_y, left_x)
        right = np.trapezoid(right_y, right_x)
        result.update({
            f"profile_{name}_min": float(values[minimum]),
            f"profile_{name}_max": float(values[maximum]),
            f"profile_{name}_range": float(np.ptp(values)),
            f"profile_{name}_s_min": float(coordinate[minimum]),
            f"profile_{name}_s_max": float(coordinate[maximum]),
            f"profile_{name}_asymmetry": float(left - right),
        })
    # Keep a stable schema so missing oxidation values remain explicit in CSV output.
    if oxidation is None:
        for suffix in ("min", "max", "range", "s_min", "s_max", "asymmetry"):
            result[f"profile_weighted_mean_oxi_{suffix}"] = np.nan
    return result, oxidation_status


def extract_source(source: str, args) -> pd.DataFrame:
    prefix = "dft" if source == "relaxed" else "bvse"
    cache = args.output_dir / f"{prefix}_descriptors.csv"
    existing = pd.DataFrame()
    if cache.exists() and not args.rebuild_descriptors:
        existing = pd.read_csv(cache)
    records = ({row["source"]: row for row in existing.to_dict("records")}
               if not existing.empty else {})
    # Successful and partial rows are complete; old all-or-nothing failures are retried.
    completed = {
        source for source, row in records.items()
        if row.get("descriptor_status") in {"ok", "partial_oxidation_missing"}
    }
    shared = SimpleNamespace(
        trajectory_source=source, target_source="em_dft",
        neighbour_cutoff=args.neighbour_cutoff,
        preprocessed_cache_dir=args.preprocessed_cache_dir,
        rebuild_preprocessed_cache=False,
    )
    items = prepare_litraj(args.data_root, "nebDFT2k", shared)["test"]
    if args.max_hops is not None:
        items = items[:args.max_hops]
    for number, item in enumerate(items, 1):
        if item.source in completed:
            continue
        row: dict[str, object] = {"source": item.source, "trajectory_source": source,
                                 "edge_reconstruction": "final Li endpoint appended to initial NEB image"}
        try:
            if source == "relaxed":
                event_features, event_oxidation_status = reconstructed_edge_features(item)
                row.update(event_features)
                row["event_oxidation_status"] = event_oxidation_status
            profile_features, profile_oxidation_status = profile_at_images(item)
            row.update({f"{prefix}_{key}": value for key, value in profile_features.items()})
            row["profile_oxidation_status"] = profile_oxidation_status
            oxidation_ok = (
                profile_oxidation_status == "ok" and
                (source != "relaxed" or row.get("event_oxidation_status") == "ok")
            )
            row["descriptor_status"] = "ok" if oxidation_ok else "partial_oxidation_missing"
        except Exception as exc:  # retain failures for an auditable denominator and resume
            row["descriptor_status"] = f"error: {type(exc).__name__}: {exc}"
        records[item.source] = row
        pd.DataFrame(records.values()).to_csv(cache.with_suffix(".tmp"), index=False)
        cache.with_suffix(".tmp").replace(cache)
        print(f"[{source}] {number}/{len(items)} {item.source}: {row['descriptor_status']}", flush=True)
    return pd.DataFrame(records.values())


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, samples: int,
                       rng: np.random.Generator) -> tuple[float, float]:
    boot = []
    for _ in range(samples):
        indices = rng.integers(0, len(x), len(x))
        boot.append(spearmanr(x[indices], y[indices]).statistic)
    return tuple(np.nanquantile(boot, [0.025, 0.975]))


def analyse(data: pd.DataFrame, args) -> pd.DataFrame:
    descriptor_columns = [c for c in data if c.startswith("litraj_") or "_profile_" in c]
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    for outcome in ("delta_pos", "delta_bvse"):
        for feature in descriptor_columns:
            subset = data[[feature, outcome]].dropna().to_numpy(float)
            if len(subset) < 20 or np.ptp(subset[:, 0]) == 0:
                continue
            result = spearmanr(subset[:, 0], subset[:, 1])
            low, high = bootstrap_spearman(subset[:, 0], subset[:, 1],
                                           args.bootstrap_samples, rng)
            rows.append({
                "family": "confirmatory_published" if feature in OFFICIAL_FEATURES else "exploratory_profile",
                "outcome": outcome, "feature": feature, "n": len(subset),
                "spearman_rho": result.statistic, "p_value": result.pvalue,
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=[
            "family", "outcome", "feature", "n", "spearman_rho", "p_value",
            "bootstrap_ci_low", "bootstrap_ci_high", "fdr_p_value",
        ])
    result["fdr_p_value"] = np.nan
    for (_, _), indices in result.groupby(["family", "outcome"]).groups.items():
        result.loc[indices, "fdr_p_value"] = bh_adjust(result.loc[indices, "p_value"].to_numpy())
    return result.sort_values(["family", "outcome", "fdr_p_value"])


def ranked_cases(data: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows = []
    for outcome in ("delta_pos", "delta_bvse"):
        ordered = data.sort_values(outcome)
        for label, group in (("largest_harm", ordered.head(top_k)),
                             ("largest_help", ordered.tail(top_k).iloc[::-1])):
            selected = group.copy()
            selected.insert(0, "effect", label)
            selected.insert(0, "outcome", outcome)
            rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def write_plan(path: Path) -> None:
    path.write_text("""LiTraj descriptor experiment hierarchy
======================================

E1 confirmatory published descriptors
  Test the five reported EdgeFeaturizer variables against delta_pos and delta_bvse.
  Interpret bootstrap intervals and FDR-adjusted p-values; do not select features post hoc.

E2 exploratory trajectory profiles
  Test the minimum, maximum, range, extremum location, and left-right asymmetry of
  Voronoi volume, effective coordination, neighbour covalent radius, and weighted
  neighbour oxidation state along DFT and BVSE paths. Treat these as hypothesis-generating.

E3 case analysis
  Inspect the largest-help and largest-harm hops using the exported table. Compare
  profile shapes and chemistry rather than inferring a mechanism from one scalar.

Validity notes
  Published definitions are run with the official ions EdgeFeaturizer, but the pristine
  edge structure is reconstructed by appending the final Li endpoint to the initial NEB
  image. This must be disclosed and should be validated on examples if pristine structures
  become available. Correlations describe associations and do not establish causality.
""")


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.per_hop_analysis)
    merged = base
    for source in args.sources:
        descriptors = extract_source(source, args)
        good = descriptors[descriptors.descriptor_status.isin(
            ["ok", "partial_oxidation_missing"]
        )].drop(
            columns=["trajectory_source", "edge_reconstruction", "descriptor_status",
                     "event_oxidation_status", "profile_oxidation_status"],
            errors="ignore")
        merged = merged.merge(good, on="source", how="left", validate="one_to_one")
    correlations = analyse(merged, args)
    merged.to_csv(args.output_dir / "per_hop_with_litraj_descriptors.csv", index=False)
    correlations.to_csv(args.output_dir / "descriptor_correlations.csv", index=False)
    ranked_cases(merged, args.top_k).to_csv(args.output_dir / "largest_help_and_harm.csv", index=False)
    write_plan(args.output_dir / "experiment_plan.txt")
    print("\nStrongest associations by analysis family:")
    display = correlations.assign(magnitude=correlations.spearman_rho.abs()).sort_values(
        ["family", "outcome", "magnitude"], ascending=[True, True, False])
    print(display.groupby(["family", "outcome"]).head(5).drop(columns="magnitude").to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
