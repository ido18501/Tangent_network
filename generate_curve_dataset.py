from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from utils.curve_generation import (
    generate_random_simple_fourier_curve,
    generate_random_simple_piecewise_curve,
)


def _generate_one_curve(task: tuple[int, int, str, int, bool, float, float, bool, float, float]) -> tuple[int, np.ndarray, float]:
    """
    task:
      (index, seed, family, num_points, simple_only, min_size, max_size,
       near_contact_check, min_sep_frac, min_gap_frac)
    """
    (
        index,
        seed,
        family,
        num_points,
        simple_only,
        min_size,
        max_size,
        near_contact_check,
        min_sep_frac,
        min_gap_frac,
    ) = task

    rng = np.random.default_rng(seed + index)
    t0 = time.perf_counter()

    if family == "fourier":
        t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
        curve, _ = generate_random_simple_fourier_curve(
            t=t,
            max_freq=5,
            scale=0.9,
            decay_power=2.0,
            rng=rng,
            max_tries=300,
            center=True,
            fit_to_canvas=True,
            min_size=min_size,
            max_size=max_size,
            enforce_simple=simple_only,
            near_contact_check=near_contact_check,
            min_separation_fraction=min_sep_frac,
            min_index_gap_fraction=min_gap_frac,
        )
    elif family == "piecewise":
        curve = generate_random_simple_piecewise_curve(
            num_points=num_points,
            rng=rng,
            closed=True,
            max_tries=300,
            fit_to_canvas=True,
            min_size=min_size,
            max_size=max_size,
            near_contact_check=near_contact_check,
            min_separation_fraction=min_sep_frac,
            min_index_gap_fraction=min_gap_frac,
        )
    else:
        raise ValueError(f"Unsupported family: {family}")

    dt = time.perf_counter() - t0
    return index, curve.astype(np.float32), dt


def _save_split(
    split_dir: Path,
    count: int,
    seed: int,
    family: str,
    num_points: int,
    simple_only: bool,
    min_size: float,
    max_size: float,
    near_contact_check: bool,
    min_sep_frac: float,
    min_gap_frac: float,
    num_workers: int,
) -> dict:
    split_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            i,
            seed,
            family,
            num_points,
            simple_only,
            min_size,
            max_size,
            near_contact_check,
            min_sep_frac,
            min_gap_frac,
        )
        for i in range(count)
    ]

    times = []
    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_generate_one_curve, task) for task in tasks]
        for done_idx, fut in enumerate(as_completed(futures), start=1):
            index, curve, dt = fut.result()
            np.save(split_dir / f"{index:06d}.npy", curve)
            times.append(dt)

            if done_idx % 100 == 0 or done_idx == count:
                print(
                    f"[{split_dir.name}] {done_idx}/{count} saved | "
                    f"avg_gen_time={np.mean(times):.4f}s | "
                    f"elapsed={time.perf_counter() - start:.1f}s"
                )

    total_elapsed = time.perf_counter() - start
    return {
        "count": count,
        "mean_generation_time_sec": float(np.mean(times)) if times else 0.0,
        "median_generation_time_sec": float(np.median(times)) if times else 0.0,
        "max_generation_time_sec": float(np.max(times)) if times else 0.0,
        "total_elapsed_sec": float(total_elapsed),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--family", type=str, default="fourier", choices=["fourier", "piecewise"])
    p.add_argument("--num-points", type=int, default=300)

    p.add_argument("--train-count", type=int, default=5000)
    p.add_argument("--val-count", type=int, default=1000)
    p.add_argument("--test-count", type=int, default=1000)

    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num-workers", type=int, default=15)

    p.add_argument("--simple-curves-only", action="store_true", default=True)
    p.add_argument("--min-size", type=float, default=0.45)
    p.add_argument("--max-size", type=float, default=0.75)
    p.add_argument("--near-contact-check", action="store_true", default=True)
    p.add_argument("--min-separation-fraction", type=float, default=0.04)
    p.add_argument("--min-index-gap-fraction", type=float, default=0.08)

    args = p.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Generating dataset under: {output_root}")
    print(f"Using {args.num_workers} workers")

    train_stats = _save_split(
        split_dir=output_root / "train",
        count=args.train_count,
        seed=args.seed + 0,
        family=args.family,
        num_points=args.num_points,
        simple_only=args.simple_curves_only,
        min_size=args.min_size,
        max_size=args.max_size,
        near_contact_check=args.near_contact_check,
        min_sep_frac=args.min_separation_fraction,
        min_gap_frac=args.min_index_gap_fraction,
        num_workers=args.num_workers,
    )

    val_stats = _save_split(
        split_dir=output_root / "val",
        count=args.val_count,
        seed=args.seed + 100000,
        family=args.family,
        num_points=args.num_points,
        simple_only=args.simple_curves_only,
        min_size=args.min_size,
        max_size=args.max_size,
        near_contact_check=args.near_contact_check,
        min_sep_frac=args.min_separation_fraction,
        min_gap_frac=args.min_index_gap_fraction,
        num_workers=args.num_workers,
    )

    test_stats = _save_split(
        split_dir=output_root / "test",
        count=args.test_count,
        seed=args.seed + 200000,
        family=args.family,
        num_points=args.num_points,
        simple_only=args.simple_curves_only,
        min_size=args.min_size,
        max_size=args.max_size,
        near_contact_check=args.near_contact_check,
        min_sep_frac=args.min_separation_fraction,
        min_gap_frac=args.min_index_gap_fraction,
        num_workers=args.num_workers,
    )

    metadata = {
        "family": args.family,
        "num_points": args.num_points,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "simple_curves_only": args.simple_curves_only,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "near_contact_check": args.near_contact_check,
        "min_separation_fraction": args.min_separation_fraction,
        "min_index_gap_fraction": args.min_index_gap_fraction,
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
    }

    with open(output_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))
    print(f"Done. Dataset written to: {output_root}")


if __name__ == "__main__":
    main()
