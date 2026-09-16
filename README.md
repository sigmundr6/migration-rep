# Learning Representations of Lithium-Ion Migration Pathways for Barrier Prediction

This research codebase studies how the geometry and local atomic environment of a lithium-ion migration pathway help predict its migration energy barrier. It uses LiTraj trajectories to compare crystal-only, pathway-aware, and ordered trajectory representations, with further experiments on bond-valence site-energy (BVSE) trajectories and density-functional theory (DFT) targets.

## Concept

Each sample consists of a crystal structure, a migration trajectory, and a target barrier in eV. A periodic crystal graph encodes atomic environments. Pathway-aware methods then connect relevant atoms through weighted hyperedges, or construct a graph along the trajectory, to describe the migration event. The model combines crystal and event representations to predict a scalar barrier.

The central question is whether information about the whole path, its ordering, and its changing environment improves prediction beyond the crystal or hop midpoint alone. Randomized and shuffled controls test which parts of that information matter. The shared implementation keeps data preparation, training, and evaluation consistent across candidate representations.

A second question concerns fidelity: how predictions change when using DFT-relaxed paths versus inexpensive BVSE paths, and whether a BVSE barrier can help predict a DFT barrier through feature fusion or residual (delta) learning. BVSE and DFT targets are kept distinct in the experiment interfaces.

## Repository structure

| Path | Purpose |
| --- | --- |
| `Proj/litraj_hypergraphs/` | Shared geometry, sample construction, models, training, and BVSE utilities. |
| `Proj/run_*.py` | Experiment entry points and comparison runners. |
| `Proj/analyze_*.py` | Statistical, descriptor, and per-hop analyses. |
| `Proj/pathway_hypergraph_representation_poc.py` | Standalone initial representation proof of concept. |
| `Proj/pathway_hypergraph_hierarchical.py` | Standalone hierarchical pathway prototype. |
| `Proj/pathway_hypergraph_representation.py` | Legacy representation comparison using the shared pipeline. |
| `Proj/generate_bvse_trajectories.py` | Generate BVSE trajectories and a dataset manifest from structures. |
| `Proj/validate_bvse_generator.py` | Compare regenerated trajectories and barriers with reference data. |
| `Proj/generate_dissertation_figures.py` | Produce PNG and PDF figures from saved experiment results. |
| `LiTraj-main/` | Included LiTraj Python utilities for downloading and loading benchmark data. |
| `results/` | Fidelity comparisons, statistical results, descriptor analyses, and figures. |
| `Proj/results/` | Material-disjoint evaluations, positional controls, sensitivity results, and external dataset audits. |
| `validation/` | Saved BVSE generator validation artifacts. |

Within `Proj/litraj_hypergraphs/`, `geometry.py` handles periodic geometry, `candidates.py` constructs representations, `data.py` loads and prepares samples, `model.py` defines the barrier predictor, and `experiment.py` provides the shared command-line and training pipeline. `bvse.py` implements BVSE optimization helpers; `test_candidates.py` checks candidate and model behavior.

Downloaded datasets, environments, caches, checkpoints, and paper drafts are excluded from Git. Scripts, this README, and retained result artifacts are included. Paths such as `data/`, `cache/`, and `checkpoints/` are created or populated locally.

## Twelve-method comparison

`Proj/run_all_hyperedge_methods.py` compares these methods:

| Method | Representation |
| --- | --- |
| `crystal` | Crystal graph baseline without pathway context. |
| `midpoint` | Local environment around the migration midpoint. |
| `trajectory` | A weighted hyperedge describing the whole path. |
| `positional` | Whole-path context with reaction-coordinate position features. |
| `segmented` | Ordered hyperedges for successive path segments. |
| `randomized` | Randomized atom-to-path assignments as a control. |
| `image_coordination` | Coordination environments at trajectory images. |
| `geometric_bottleneck` | Local geometric bottleneck environments. |
| `coordination_change` | Atoms associated with changes in coordination along the path. |
| `species_image` | Species-typed environments at trajectory images. |
| `position_aware_trajectory_graph` | A position-aware graph along the trajectory. |
| `position_aware_trajectory_graph_shuffled` | A shuffled-order trajectory graph control. |

The candidate registry also contains the `bottleneck` alias and `positional_shuffled_s` control used outside this default twelve-method comparison.

## Setup

Run the following commands from the repository root. Use a Python environment compatible with PyTorch and the scientific packages below; the repository does not currently include a pinned environment lockfile.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy pandas scipy ase requests tqdm matplotlib scikit-learn
export PYTHONPATH="$PWD/LiTraj-main:$PWD/Proj${PYTHONPATH:+:$PYTHONPATH}"
```

The `PYTHONPATH` setting makes the included LiTraj utilities and project modules importable without installing the bundled package. Repeat the activation and export in new terminal sessions.

For BVSE trajectory generation and validation, also install the dependency version specified by the code:

```bash
python -m pip install 'ions==0.4.1'
```

The shared training CLI supports `--device cpu`, `--device mps`, and `--device auto`. Automatic selection uses Apple MPS when available and otherwise CPU.

## Running experiments

### Synthetic checks

The comparison runner uses synthetic data when no trajectory, manifest, or LiTraj dataset is supplied. This provides a small workflow check without downloading the benchmark:

```bash
python Proj/run_all_hyperedge_methods.py --device cpu --epochs 2 --synthetic-samples 24
python -m unittest litraj_hypergraphs.test_candidates
```

Synthetic checks exercise the implementation; benchmark conclusions should come from the real-data evaluations.

### LiTraj benchmark

Download `nebDFT2k` into the ignored local data directory:

```bash
mkdir -p data
python -c "from litraj.data import download_dataset; download_dataset('nebDFT2k', 'data')"
```

Check sample construction and run the twelve-method comparison:

```bash
python Proj/sanity_check_litraj_hyperedges.py --data-root data
python Proj/run_all_hyperedge_methods.py \
  --litraj-data data --trajectory-source relaxed --target-source em_dft \
  --device cpu --epochs 35 --seed 7
```

Use `--methods crystal midpoint trajectory positional` to select a smaller comparison. The shared pipeline supports preprocessing caches, checkpoint saving, and `--resume`; use distinct checkpoint directories for separate configurations.

### Custom trajectories and BVSE generation

A custom manifest uses columns `path,target,split` and optionally `migrating_index`. Paths point to multi-frame extended XYZ files; relative paths are resolved against the manifest directory. Splits are `train`, `val`, or `test`.

```bash
python Proj/run_all_hyperedge_methods.py --manifest data/custom/manifest.csv --device cpu
python Proj/generate_bvse_trajectories.py data/structures/example.cif \
  --output-dir data/bvse_generated --manifest data/bvse_generated/manifest.csv
python Proj/validate_bvse_generator.py --data-root data --samples 5 --write-trajectories
```

Replace the example structure and manifest paths with actual inputs. The generator assigns splits by source material rather than by individual hop. `Proj/run_external_bvse_pipeline.sh` combines generation, dataset auditing, and external evaluation; its `--help` describes pilot and audit-only modes.

### Fidelity, controls, and analysis

| Scripts | Question or output |
| --- | --- |
| `run_nebdft2k_bvse_dft_comparison.py`, `run_cross_fidelity_experiment.py` | Compare trajectory/target fidelities, including BVSE paths for DFT-barrier prediction. |
| `run_exploratory_statistical_evaluation.py`, `run_final_statistical_evaluation.py` | Repeated evaluations, predictions, and paired statistical comparisons. |
| `run_material_disjoint_evaluation.py` | Evaluate generalization across separated materials. |
| `run_positional_controls.py`, `run_random_reversal_augmentation.py`, `analyze_positional_controls.py` | Test position, path direction, and reversal effects. |
| `run_pathway_hyperparameter_sensitivity.py`, `plot_pathway_sensitivity.py` | Measure and plot sensitivity to pathway construction choices. |
| `analyze_per_hop_pathway_benefit.py`, `analyze_litraj_classical_profiles.py` | Relate per-hop performance changes to geometric and classical descriptors. |
| `run_multivariate_descriptor_experiment.py` | Evaluate combinations of explanatory descriptors. |
| `analyze_seed_reliability_and_ensembles.py` | Analyze split/seed variation and ensemble behavior. |
| `run_external_bvse_generalization.py` | Audit external data and evaluate BVSE generalization. |

These scripts live under `Proj/`. Check each script's `--help` for its input paths and configuration. Run the BVSE/DFT comparison before cross-fidelity analysis, which consumes its reference CSV. Saved outputs are retained in both result directories because different experiment runners use different defaults.

To regenerate the main dissertation figures from the retained tables:

```bash
python Proj/generate_dissertation_figures.py
```

## Reproducibility

The saved CSV/JSON tables include predictions, metrics, split assignments, or analysis settings depending on the experiment. Figures and a small generator validation sample are also retained. Full benchmark data and trained checkpoints must be downloaded or regenerated separately.

For a new experiment, record the command, seed, split strategy, geometry source, target source, and dependency versions, and use a separate output directory where supported. The quick-start commands are examples, not a claim that they reproduce every retained result with identical settings. Inspect each runner's arguments and the corresponding saved artifacts when reproducing a particular experiment.

## LiTraj acknowledgement and license

This project uses the [LiTraj dataset and Python utilities](https://github.com/AIRI-Institute/LiTraj), particularly the `nebDFT2k` benchmark, for lithium-ion migration pathway and barrier experiments. The utilities included under `LiTraj-main/` originate from LiTraj; credit for that software and the benchmark data belongs to their original authors.

LiTraj's software is distributed under the [MIT License](https://github.com/AIRI-Institute/LiTraj/blob/main/LICENSE), copyright (c) 2024 AI Research and Skoltech. We use and redistribute the included LiTraj code under those terms and retain the original copyright and permission notice in [LiTraj-main/LICENSE](LiTraj-main/LICENSE). This notice applies to the included LiTraj software; it does not assign a license to this project's original code.

Please cite the LiTraj paper when using its datasets:

Dembitskiy, A. D. et al. (2025). [Benchmarking machine learning models for predicting lithium ion migration](https://doi.org/10.1038/s41524-025-01571-z). *npj Computational Materials*, 11, 131.

LiTraj also credits the [Materials Project](https://next-gen.materialsproject.org/) as the source of its crystal structures and identifies their license as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve the applicable source attribution and terms when reusing or redistributing those data. Full downloaded benchmark datasets are excluded from this repository; retained results and validation artifacts are described above.
