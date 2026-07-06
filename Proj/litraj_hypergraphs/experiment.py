"""Common command-line and training harness for all four candidates."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .candidates import CANDIDATES, HypergraphSample
from .data import (
    build_from_arrays,
    load_litraj,
    load_manifest,
    materialize_prepared,
    prepare_manifest,
    prepare_litraj,
    read_extxyz,
    synthetic_dataset,
)
from .model import CandidateBarrierModel


def parser_for(candidate: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"LiTraj hyperedge experiment using the {candidate!r} construction."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--trajectory", type=Path, help="multi-image extended XYZ file")
    source.add_argument("--manifest", type=Path, help="CSV: path,target,split[,migrating_index]")
    source.add_argument("--litraj-data", type=Path, help="root consumed by litraj.data.load_data")
    parser.add_argument(
        "--litraj-dataset", choices=["nebDFT2k"], default="nebDFT2k",
        help="official full-trajectory dataset (use --manifest for other trajectory sources)",
    )
    parser.add_argument(
        "--trajectory-source", choices=["relaxed", "init"], default="relaxed",
        help="nebDFT2k geometry: DFT-relaxed or BVSE-NEB initial trajectory",
    )
    parser.add_argument(
        "--target-source", choices=["auto", "em_dft", "em_bvse"], default="auto",
        help="nebDFT2k target; auto pairs relaxed/em_dft and init/em_bvse",
    )
    parser.add_argument("--target", type=float, help="barrier for --trajectory")
    parser.add_argument("--migrating-index", type=int)
    parser.add_argument("--neighbour-cutoff", type=float, default=3.5)
    parser.add_argument("--hyperedge-cutoff", type=float, default=3.0)
    parser.add_argument("--hyperedge-sigma", type=float, default=1.5)
    parser.add_argument("--bottleneck-neighbours", type=int, default=4)
    parser.add_argument("--change-top-k", type=int, default=8)
    parser.add_argument("--num-segments", type=int, default=5)
    parser.add_argument("--segment-sigma", type=float, default=0.18)
    parser.add_argument(
        "--presence-threshold", type=float, default=1e-6,
        help="minimum summed incidence for a hyperedge to enter temporal processing",
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="stop after this many epochs without a meaningful validation improvement; 0 disables",
    )
    parser.add_argument(
        "--early-stopping-min-delta", type=float, default=0.0,
        help="minimum validation-MAE improvement that resets early-stopping patience",
    )
    parser.add_argument(
        "--auxiliary-mode", choices=["none", "fusion", "delta"], default="none",
        help="combine the trajectory with em_bvse directly or by calibrated delta learning",
    )
    parser.add_argument("--synthetic-samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--reversal-invariant",
        action="store_true",
        help="average forward/reverse path encodings; off by default because LiTraj barrier directionality must be verified",
    )
    parser.add_argument(
        "--random-reversal-augmentation",
        action="store_true",
        help=(
            "for positional training samples, independently replace s by 1-s "
            "with probability 0.5; validation and test samples are unchanged"
        ),
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps"], default="auto",
        help="compute device; auto uses Apple MPS when available, otherwise CPU",
    )
    parser.add_argument(
        "--mps-data", choices=["stream", "resident"], default="stream",
        help="stream one sample to MPS by default; resident is faster only if the full dataset fits safely",
    )
    parser.add_argument(
        "--mps-memory-report", action="store_true",
        help="report PyTorch and Metal allocator memory at checkpoint epochs",
    )
    parser.add_argument(
        "--preprocessed-cache-dir", type=Path, default=Path("cache/litraj_preprocessed"),
        help="cache shared trajectory graphs/path geometry for subsequent runs",
    )
    parser.add_argument(
        "--rebuild-preprocessed-cache", action="store_true",
        help="ignore and replace a matching preprocessing cache",
    )
    parser.add_argument(
        "--mps-isolate-methods",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "run each comparison method in a fresh process on MPS (default: on); "
            "this bounds Metal/MPSGraph allocations that empty_cache cannot reclaim"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/litraj_hypergraphs"),
        help="directory for per-method training checkpoints",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=5,
        help="save and report every N epochs (default: 5; zero disables periodic saves)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume each method from its checkpoint in --checkpoint-dir",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=sorted(CANDIDATES),
        help="comparison runner only: restrict execution to these representations",
    )
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        built = torch.backends.mps.is_built()
        raise SystemExit(
            "--device mps was requested, but MPS is unavailable "
            f"(PyTorch MPS support built={built}). Use --device cpu or an "
            "Apple-Silicon/macOS environment with MPS enabled."
        )
    return torch.device(requested)


def move_splits(splits, device: torch.device):
    return {
        name: [sample.to(device) for sample in samples]
        for name, samples in splits.items()
    }


def release_device_cache(device: torch.device, run_name: str) -> None:
    """Release method-local accelerator caches before constructing the next model."""
    if device.type != "mps":
        return
    torch.mps.synchronize()
    before = torch.mps.driver_allocated_memory()
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()
    after = torch.mps.driver_allocated_memory()
    print(
        f"[{run_name}] released MPS method cache: "
        f"{before / 2**20:.1f} -> {after / 2**20:.1f} MiB driver memory",
        flush=True,
    )


def _for_model(sample: HypergraphSample, device: torch.device) -> HypergraphSample:
    return sample if sample.target.device == device else sample.to(device)


def mae(
    model: CandidateBarrierModel, samples: list[HypergraphSample], device: torch.device
) -> float:
    model.eval()
    with torch.no_grad():
        errors = []
        for stored_sample in samples:
            sample = _for_model(stored_sample, device)
            errors.append(abs(model(sample).item() - sample.target.item()))
            if sample is not stored_sample:
                del sample
    return float(np.mean(errors)) if errors else math.nan


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def checkpoint_path(args, run_name: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in run_name)
    return args.checkpoint_dir / f"{safe_name}_seed{args.seed}.pt"


def dataset_run_name(candidate: str, args) -> str:
    if args.litraj_data and (
        args.trajectory_source != "relaxed" or args.target_source not in {"auto", "em_dft"}
    ):
        target = "em_bvse" if args.target_source == "auto" else args.target_source
        name = f"{args.trajectory_source}_{target}_{candidate}"
    else:
        name = candidate
    return name if args.auxiliary_mode == "none" else f"{name}_{args.auxiliary_mode}"


def save_checkpoint(
    path: Path, *, epoch: int, model, optimizer, best_state, best_val: float,
    order_rng: random.Random, history: list[dict[str, float]], args,
    best_epoch: int = 0, epochs_without_improvement: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "epoch": epoch,
        "model_state": _cpu_state_dict(model),
        "optimizer_state": _to_cpu(optimizer.state_dict()),
        "best_state": {key: value.cpu() for key, value in best_state.items()} if best_state else None,
        "best_val": best_val,
        "order_rng_state": order_rng.getstate(),
        "history": history,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "configuration": {
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "learning_rate": args.learning_rate,
            "reversal_invariant": args.reversal_invariant,
            "random_reversal_augmentation": getattr(
                args, "random_reversal_augmentation", False
            ),
            "presence_threshold": args.presence_threshold,
            "seed": args.seed,
            "auxiliary_mode": args.auxiliary_mode,
            "delta_slope": getattr(args, "delta_slope", 1.0),
            "delta_intercept": getattr(args, "delta_intercept", 0.0),
            "early_stopping_patience": getattr(args, "early_stopping_patience", 0),
            "early_stopping_min_delta": getattr(args, "early_stopping_min_delta", 0.0),
        },
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(splits, args, device: torch.device, run_name: str):
    torch.manual_seed(args.seed)
    model = CandidateBarrierModel(
        args.hidden_dim, args.layers, reversal_invariant=args.reversal_invariant,
        presence_threshold=args.presence_threshold,
        auxiliary_mode=args.auxiliary_mode,
        delta_slope=getattr(args, "delta_slope", 1.0),
        delta_intercept=getattr(args, "delta_intercept", 0.0),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_state, best_val = None, float("inf")
    order_rng = random.Random(args.seed)
    history: list[dict[str, float]] = []
    start_epoch = 0
    best_epoch = 0
    epochs_without_improvement = 0
    path = checkpoint_path(args, run_name)
    if args.resume:
        if not path.exists():
            print(f"[{run_name}] no checkpoint at {path}; starting from epoch 0", flush=True)
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            expected = checkpoint.get("configuration", {})
            current = {
                "hidden_dim": args.hidden_dim,
                "layers": args.layers,
                "learning_rate": args.learning_rate,
                "reversal_invariant": args.reversal_invariant,
                "random_reversal_augmentation": getattr(
                    args, "random_reversal_augmentation", False
                ),
                "presence_threshold": args.presence_threshold,
                "seed": args.seed,
                "auxiliary_mode": args.auxiliary_mode,
                "delta_slope": getattr(args, "delta_slope", 1.0),
                "delta_intercept": getattr(args, "delta_intercept", 0.0),
                "early_stopping_patience": getattr(args, "early_stopping_patience", 0),
                "early_stopping_min_delta": getattr(args, "early_stopping_min_delta", 0.0),
            }
            mismatches = {
                key: (saved, current.get(key)) for key, saved in expected.items()
                if current.get(key) != saved
            }
            if mismatches:
                raise ValueError(
                    f"checkpoint configuration mismatch for {run_name}: "
                    f"mismatches={mismatches}"
                )
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            _optimizer_to(optimizer, device)
            best_state = checkpoint["best_state"]
            best_val = float(checkpoint["best_val"])
            order_rng.setstate(checkpoint["order_rng_state"])
            history = list(checkpoint.get("history", []))
            start_epoch = int(checkpoint["epoch"])
            best_epoch = int(checkpoint.get(
                "best_epoch",
                min(history, key=lambda row: row["val_mae"])["epoch"] if history else 0,
            ))
            epochs_without_improvement = int(checkpoint.get(
                "epochs_without_improvement", max(0, start_epoch - best_epoch)
            ))
            print(f"[{run_name}] resumed from {path} at epoch {start_epoch}", flush=True)

    patience = max(0, int(getattr(args, "early_stopping_patience", 0)))
    min_delta = max(0.0, float(getattr(args, "early_stopping_min_delta", 0.0)))
    if patience and epochs_without_improvement >= patience:
        print(
            f"[{run_name}] checkpoint already early-stopped at epoch {start_epoch} "
            f"(best epoch {best_epoch})", flush=True,
        )
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, {name: mae(model, values, device) for name, values in splits.items()}

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        order = list(range(len(splits["train"])))
        order_rng.shuffle(order)
        absolute_errors = []
        for index in order:
            sample = _for_model(splits["train"][index], device)
            augmented_sample = False
            if (
                getattr(args, "random_reversal_augmentation", False)
                and str(sample.metadata["candidate"]) == "positional"
                and order_rng.random() < 0.5
            ):
                sample = replace(sample, path_position=1.0 - sample.path_position)
                augmented_sample = True
            optimizer.zero_grad(set_to_none=True)
            prediction = model(sample)
            loss = (prediction - sample.target).square()
            loss.backward()
            optimizer.step()
            absolute_errors.append(abs(prediction.detach().item() - sample.target.item()))
            # Drop the final references promptly. This does not repair driver-side
            # MPSGraph caching, but prevents live autograd tensors from spanning
            # iterations while samples are streamed from CPU.
            del loss, prediction
            if augmented_sample or sample is not splits["train"][index]:
                del sample
        validation = mae(model, splits["val"], device)
        previous_best = best_val
        if validation < best_val:
            best_val = validation
            best_state = _cpu_state_dict(model)
            best_epoch = epoch
        if validation < previous_best - min_delta:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        train_mae = float(np.mean(absolute_errors))
        history.append({"epoch": float(epoch), "train_mae": train_mae, "val_mae": validation})
        report = args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0
        final_epoch = epoch == args.epochs
        early_stop = patience > 0 and epochs_without_improvement >= patience
        if report or final_epoch or early_stop:
            print(
                f"[{run_name}] epoch {epoch:>3}/{args.epochs}: "
                f"train_MAE={train_mae:.4f} eV  val_MAE={validation:.4f} eV  "
                f"best_val={best_val:.4f} eV",
                flush=True,
            )
            save_checkpoint(
                path, epoch=epoch, model=model, optimizer=optimizer,
                best_state=best_state, best_val=best_val, order_rng=order_rng,
                history=history, args=args, best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
            )
            if device.type == "mps" and args.mps_memory_report:
                print(
                    f"[{run_name}] MPS memory: tensors="
                    f"{torch.mps.current_allocated_memory() / 2**20:.1f} MiB  driver="
                    f"{torch.mps.driver_allocated_memory() / 2**20:.1f} MiB  recommended="
                    f"{torch.mps.recommended_max_memory() / 2**20:.1f} MiB",
                    flush=True,
                )
        if early_stop:
            print(
                f"[{run_name}] early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, best_val={best_val:.4f} eV",
                flush=True,
            )
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {name: mae(model, values, device) for name, values in splits.items()}


def describe(sample: HypergraphSample) -> None:
    memberships = int((sample.incidence > 0).sum())
    print(f"source: {sample.source}")
    print(f"candidate: {sample.metadata['candidate']}")
    print(f"atoms/images/hyperedges: {len(sample.atomic_numbers)}/{len(sample.frames)}/{sample.incidence.shape[1]}")
    print(f"nonzero memberships: {memberships}")
    print(f"metadata: {sample.metadata}")


def main(candidate: str) -> None:
    if candidate not in CANDIDATES:
        raise ValueError(candidate)
    args = parser_for(candidate).parse_args()
    device = resolve_device(args.device)
    print(f"device: {device}")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.trajectory:
        if args.target is None:
            raise SystemExit("--trajectory requires --target")
        numbers, frames, cell, index = read_extxyz(args.trajectory, args.migrating_index)
        sample = build_from_arrays(
            candidate, numbers, frames, cell, index, args.target, str(args.trajectory), args
        ).to(device)
        describe(sample)
        model = CandidateBarrierModel(
            args.hidden_dim, args.layers, reversal_invariant=args.reversal_invariant,
            presence_threshold=args.presence_threshold,
        ).to(device)
        prediction = model(sample)
        (prediction - sample.target).square().backward()
        print(f"forward/backward: PASS; untrained prediction={prediction.item():.4f} eV")
        return
    if args.manifest:
        splits = load_manifest(args.manifest, candidate, args)
    elif args.litraj_data:
        splits = load_litraj(args.litraj_data, args.litraj_dataset, candidate, args)
    else:
        splits = synthetic_dataset(candidate, args.synthetic_samples, args.seed, args)
    storage_device = torch.device("cpu") if device.type == "mps" and args.mps_data == "stream" else device
    splits = move_splits(splits, storage_device)
    print(f"candidate: {candidate}; splits: { {k: len(v) for k, v in splits.items()} }")
    describe(splits["train"][0])
    model, metrics = train(splits, args, device, dataset_run_name(candidate, args))
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"parameters: {count}")
    print("  ".join(f"{key}_mae={value:.4f} eV" for key, value in metrics.items()))


def comparison_main(candidates: tuple[str, ...]) -> None:
    """Run several incidence constructions through the identical shared harness."""
    args = parser_for("legacy comparison").parse_args()
    if args.methods:
        requested = set(args.methods)
        candidates = tuple(candidate for candidate in candidates if candidate in requested)
    device = resolve_device(args.device)
    print(f"device: {device}")

    # MPSGraph and Metal may retain shape-specialized resources outside
    # torch.mps.current_allocated_memory().  They are not reliably released by
    # empty_cache(), but macOS releases them when the process exits.  Keep each
    # candidate in a separate process so a long comparison cannot accumulate
    # another candidate's driver allocations.
    isolated_candidate = os.environ.get("LITRAJ_ISOLATED_CANDIDATE")
    if device.type == "mps" and args.mps_isolate_methods and not isolated_candidate:
        for candidate in candidates:
            environment = os.environ.copy()
            environment["LITRAJ_ISOLATED_CANDIDATE"] = candidate
            print(f"\n=== isolated MPS method: {candidate} ===", flush=True)
            completed = subprocess.run([sys.executable, "-u", *sys.argv], env=environment)
            if completed.returncode != 0:
                raise SystemExit(
                    f"MPS method {candidate!r} failed with exit code {completed.returncode}"
                )
        return
    if isolated_candidate:
        if isolated_candidate not in candidates:
            raise SystemExit(f"unknown isolated candidate: {isolated_candidate}")
        candidates = (isolated_candidate,)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    prepared_litraj = None
    prepared_manifest = None
    if args.litraj_data:
        prepared_litraj = prepare_litraj(args.litraj_data, args.litraj_dataset, args)
    elif args.manifest:
        prepared_manifest = prepare_manifest(args.manifest, args)

    if args.auxiliary_mode in {"fusion", "delta"}:
        prepared = prepared_litraj if prepared_litraj is not None else prepared_manifest
        if prepared is None:
            raise SystemExit("--auxiliary-mode requires LiTraj data or a manifest with em_bvse")
        train_pairs = [
            (item.cheap_barrier, item.target) for item in prepared["train"]
            if item.cheap_barrier is not None
        ]
        if len(train_pairs) != len(prepared["train"]):
            raise SystemExit("every sample needs em_bvse for scalar fusion/delta learning")
        cheap, target = np.asarray(train_pairs, dtype=float).T
        args.delta_slope, args.delta_intercept = np.polyfit(cheap, target, 1)
        print(
            f"train-only BVSE calibration: DFT={args.delta_slope:.6f}*BVSE"
            f"{args.delta_intercept:+.6f}", flush=True,
        )

    def get_splits(candidate: str):
        if args.manifest:
            return materialize_prepared(prepared_manifest, candidate, args)
        if args.litraj_data:
            return materialize_prepared(prepared_litraj, candidate, args)
        return synthetic_dataset(candidate, args.synthetic_samples, args.seed, args)

    if args.trajectory:
        if args.target is None:
            raise SystemExit("--trajectory requires --target")
        numbers, frames, cell, index = read_extxyz(args.trajectory, args.migrating_index)
        for candidate in candidates:
            sample = build_from_arrays(
                candidate, numbers, frames, cell, index, args.target, str(args.trajectory), args
            ).to(device)
            model = CandidateBarrierModel(
                args.hidden_dim, args.layers, reversal_invariant=args.reversal_invariant,
                presence_threshold=args.presence_threshold,
            ).to(device)
            prediction = model(sample)
            (prediction - sample.target).square().backward()
            print(f"{candidate:<20} forward/backward PASS  prediction={prediction.item():.4f} eV")
        return

    print("mode                  params    train_MAE   val_MAE    test_MAE")
    print("--------------------  --------  ----------  ---------  ---------")
    for candidate in candidates:
        storage_device = torch.device("cpu") if device.type == "mps" and args.mps_data == "stream" else device
        splits = move_splits(get_splits(candidate), storage_device)
        model, metrics = train(splits, args, device, dataset_run_name(candidate, args))
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print(
            f"{candidate:<20}  {count:>8d}  {metrics['train']:.4f} eV   "
            f"{metrics['val']:.4f} eV  {metrics['test']:.4f} eV"
        )
        del model, splits
        release_device_cache(device, candidate)
