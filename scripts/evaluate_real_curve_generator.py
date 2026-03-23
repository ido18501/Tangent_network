from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.patch_sampling import sample_patch_around_index
from models.tangent_model import TangentOperatorModel


Array = np.ndarray


def normalize(v: Array, eps: float = 1e-12) -> Array:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n <= eps:
        return np.zeros_like(v, dtype=np.float64)
    return v / n


def angle_deg(a: Array, b: Array, sign_invariant: bool = True) -> float:
    a = normalize(a)
    b = normalize(b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sign_invariant:
        c = abs(c)
    return float(np.degrees(np.arccos(c)))


def angular_error_stats(errors: list[float]) -> dict[str, float]:
    arr = np.asarray(errors, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


@dataclass
class CurveSample:
    dense_curve: Array
    image_curve: Array
    closed: bool
    score: float
    image_id: str
    metadata: dict


def _first_existing_key(data, candidates):
    for k in candidates:
        if k in data.files:
            return k
    return None


def load_samples_from_npz(path: str | Path) -> list[CurveSample]:
    data = np.load(path, allow_pickle=False)
    samples: list[CurveSample] = []

    num_curves = None
    for key in ("num_curves", "num_samples", "num_contours"):
        if key in data.files:
            num_curves = int(np.asarray(data[key]).reshape(-1)[0])
            break

    if num_curves is None:
        prefixes = sorted(
            {
                k.split("_dense")[0]
                for k in data.files
                if "_dense" in k
            }
        )
        num_curves = len(prefixes)

    for i in range(num_curves):
        dense_key = _first_existing_key(
            data,
            [
                f"curve_{i}_dense",
                f"curve_{i}_canonical",
                f"curve_{i}_canonical_points",
                f"curve_{i}_dense_curve",
                f"curve_{i}",
            ],
        )

        image_key = _first_existing_key(
            data,
            [
                f"curve_{i}_image",
                f"curve_{i}_image_points",
                f"curve_{i}_raw",
                f"curve_{i}_raw_points",
            ],
        )

        closed_key = _first_existing_key(
            data,
            [
                f"curve_{i}_closed",
            ],
        )

        score_key = _first_existing_key(
            data,
            [
                f"curve_{i}_score",
            ],
        )

        if dense_key is None:
            continue

        dense_curve = np.asarray(data[dense_key], dtype=np.float64)

        if image_key is not None:
            image_curve = np.asarray(data[image_key], dtype=np.float64)
        else:
            # fallback only if no image-space version exists
            image_curve = dense_curve.copy()

        closed = bool(np.asarray(data[closed_key]).reshape(-1)[0]) if closed_key is not None else True
        score = float(np.asarray(data[score_key]).reshape(-1)[0]) if score_key is not None else 0.0

        if dense_curve.ndim == 2 and dense_curve.shape[1] == 2 and len(dense_curve) >= 3:
            samples.append(
                CurveSample(
                    dense_curve=dense_curve,
                    image_curve=image_curve,
                    closed=closed,
                    score=score,
                    image_id=Path(path).stem,
                    metadata={
                        "dense_key": dense_key,
                        "image_key": image_key,
                    },
                )
            )

    if not samples:
        for key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim == 2 and arr.shape[1] == 2:
                samples.append(
                    CurveSample(
                        dense_curve=arr.astype(np.float64),
                        image_curve=arr.astype(np.float64),
                        closed=True,
                        score=0.0,
                        image_id=Path(path).stem,
                        metadata={"dense_key": key, "image_key": key},
                    )
                )

    return samples

def map_index_between_curves(src_idx: int, src_len: int, dst_len: int, closed: bool) -> int:
    if src_len <= 1 or dst_len <= 1:
        return 0

    if closed:
        t = float(src_idx % src_len) / float(src_len)
        return int(round(t * dst_len)) % dst_len
    else:
        t = float(src_idx) / float(max(src_len - 1, 1))
        return int(round(t * max(dst_len - 1, 1)))


def iter_curve_files(curve_dir: str | Path):
    curve_dir = Path(curve_dir)
    for p in sorted(curve_dir.glob("*.npz")):
        yield p


def fd_tangent(curve: Array, idx: int, closed: bool = True) -> Array:
    n = len(curve)

    def map_idx(i: int) -> int:
        if closed:
            return i % n
        return min(max(i, 0), n - 1)

    return np.asarray(curve[map_idx(idx + 1)] - curve[map_idx(idx - 1)], dtype=np.float64)


def build_model(patch_size: int) -> torch.nn.Module:
    return TangentOperatorModel(
        patch_size=patch_size,
        point_dim=2,
        point_mlp_dims=[64, 64, 128],
        head_dims=[128, 64],
        use_batchnorm=True,
        point_dropout=0.0,
        head_dropout=0.0,
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def extract_vector(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        if "vector" in out:
            out = out["vector"]
        elif "output" in out:
            out = out["output"]
        else:
            raise RuntimeError(f"Unsupported model dict output keys: {list(out.keys())}")
    if not torch.is_tensor(out):
        raise RuntimeError(f"Unsupported model output type: {type(out)}")
    return out


def maybe_get_weights(model: torch.nn.Module, patch_tensor: torch.Tensor) -> Array | None:
    with torch.no_grad():
        if hasattr(model, "forward_with_weights"):
            out = model.forward_with_weights(patch_tensor)
            if isinstance(out, tuple) and len(out) >= 2:
                return np.asarray(out[1].detach().cpu())
        if hasattr(model, "get_patch_weights"):
            return np.asarray(model.get_patch_weights(patch_tensor).detach().cpu())
        out = model(patch_tensor)
        if isinstance(out, dict) and "weights" in out:
            return np.asarray(out["weights"].detach().cpu())
    return None


def predict(model: torch.nn.Module, patch_centered: Array, device: torch.device) -> tuple[Array, Array | None]:
    x = torch.as_tensor(patch_centered, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        vec = np.asarray(extract_vector(model(x)).squeeze(0).detach().cpu(), dtype=np.float64)
    weights = maybe_get_weights(model, x)
    if weights is not None:
        weights = np.asarray(weights)[0].reshape(-1)
    return vec, weights


def rotation_matrix(theta_deg: float) -> Array:
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def draw_arrow(ax, origin: Array, vec: Array, scale: float, color: str, label: str | None = None, lw: float = 2.0):
    v = normalize(vec) * scale
    ax.arrow(
        float(origin[0]),
        float(origin[1]),
        float(v[0]),
        float(v[1]),
        width=0.0,
        head_width=0.03 * scale,
        head_length=0.06 * scale,
        length_includes_head=True,
        color=color,
        linewidth=lw,
    )
    if label is not None:
        ax.text(float(origin[0] + v[0]), float(origin[1] + v[1]), label, color=color, fontsize=9)


def draw_arrow_image(ax, origin: Array, vec: Array, color: str, label: str | None = None, scale_px: float = 35.0):
    v = normalize(vec) * scale_px
    ax.arrow(
        float(origin[0]),
        float(origin[1]),
        float(v[0]),
        float(v[1]),
        width=0.2,
        head_width=5.0,
        head_length=7.0,
        length_includes_head=True,
        color=color,
        linewidth=2.0,
    )
    if label is not None:
        ax.text(float(origin[0] + v[0]), float(origin[1] + v[1]), label, color=color, fontsize=9)


def match_image_path(image_dir: Path, curve_file: Path) -> Path | None:
    stem = curve_file.stem
    if stem.endswith("_curves"):
        stem = stem[:-7]
    for ext in [".jpg", ".png", ".jpeg", ".bmp", ".webp"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def save_overlay_figure(
    image: Array | None,
    image_curve: Array,
    dense_curve: Array,
    patch_points_dense: Array,
    center_idx: int,
    closed: bool,
    pred_vec: Array,
    gt_vec: Array,
    weights: Array | None,
    transformed_patch: Array,
    transformed_pred: Array,
    transformed_gt: Array,
    transform_angle_deg: float,
    title: str,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1) image overlay
    ax = axes[0]
    if image is not None:
        ax.imshow(image)
    if (
            image_curve.ndim == 2
            and image_curve.shape[1] == 2
            and np.max(np.abs(image_curve[:, 0])) > 5
            and np.max(np.abs(image_curve[:, 1])) > 5
    ):
        ax.plot(image_curve[:, 0], image_curve[:, 1], linewidth=2.0, color="deepskyblue")
    img_center_idx = map_index_between_curves(
        src_idx=int(center_idx),
        src_len=len(dense_curve),
        dst_len=len(image_curve),
        closed=closed,
    )
    center_point_img = image_curve[img_center_idx]
    ax.scatter([center_point_img[0]], [center_point_img[1]], c="yellow", s=40, marker="x")
    draw_arrow_image(ax, center_point_img, gt_vec, color="lime", label="gt")
    draw_arrow_image(ax, center_point_img, pred_vec, color="red", label="pred")
    ax.set_title("image overlay")
    ax.set_aspect("equal")
    ax.invert_yaxis() if image is None else None

    # 2) local patch
    ax = axes[1]
    centered = patch_points_dense - dense_curve[int(center_idx) % len(dense_curve)]
    ax.plot(centered[:, 0], centered[:, 1], marker="o", color="steelblue")
    ax.scatter([0.0], [0.0], c="black", marker="x", s=60)
    patch_scale = max(np.ptp(centered[:, 0]), np.ptp(centered[:, 1]), 1e-3)
    draw_arrow(ax, np.array([0.0, 0.0]), gt_vec, scale=0.8 * patch_scale, color="lime", label="gt")
    draw_arrow(ax, np.array([0.0, 0.0]), pred_vec, scale=0.8 * patch_scale, color="red", label="pred")
    ax.set_title("local patch")
    ax.set_aspect("equal")

    # 3) transformed patch comparison
    ax = axes[2]
    ax.plot(transformed_patch[:, 0], transformed_patch[:, 1], marker="o", color="mediumpurple")
    ax.scatter([0.0], [0.0], c="black", marker="x", s=60)
    tscale = max(np.ptp(transformed_patch[:, 0]), np.ptp(transformed_patch[:, 1]), 1e-3)
    draw_arrow(ax, np.array([0.0, 0.0]), transformed_gt, scale=0.8 * tscale, color="lime", label="gt_rot")
    draw_arrow(ax, np.array([0.0, 0.0]), transformed_pred, scale=0.8 * tscale, color="red", label="pred_rot")
    ax.set_title(f"rotated patch ({transform_angle_deg:.1f}°)")
    ax.set_aspect("equal")

    # 4) weights
    ax = axes[3]
    if weights is None:
        ax.text(0.5, 0.5, "weights unavailable", ha="center", va="center")
    else:
        ax.plot(np.arange(len(weights)), weights, marker="o")
    ax.set_title("learned weights")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


@dataclass
class PatchRecord:
    model_name: str
    image_file: str
    curve_file: str
    curve_index: int
    patch_center_index: int
    angular_error_deg: float
    cosine: float
    pred_norm: float
    gt_norm: float
    rotated_angular_error_deg: float
    transform_angle_deg: float


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve-dir", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--half-width", type=int, default=12)
    parser.add_argument("--patches-per-curve", type=int, default=3)
    parser.add_argument("--max-curves", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default="outputs/real_curve_eval")
    parser.add_argument("--save-num-visuals", type=int, default=24)
    parser.add_argument("--transform-angle-deg", type=float, default=35.0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = Path(args.output_dir)
    vis_dir = out_root / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, torch.nn.Module] = {}
    for item in args.checkpoints:
        if "=" in item:
            name, ckpt = item.split("=", 1)
        else:
            ckpt = item
            name = Path(ckpt).parent.name or Path(ckpt).stem
        models[name] = load_checkpoint(build_model(args.patch_size), ckpt, device)

    curve_files = list(iter_curve_files(args.curve_dir))
    if not curve_files:
        raise RuntimeError(f"No .npz files found in {args.curve_dir}")

    all_curve_items: list[tuple[Path, int, CurveSample]] = []
    for cf in curve_files:
        samples = load_samples_from_npz(cf)

        for i, s in enumerate(samples):
            print(
                cf.name,
                "dense_key=", s.metadata.get("dense_key"),
                "image_key=", s.metadata.get("image_key"),
                "dense_range=", s.dense_curve.min(), s.dense_curve.max(),
                "image_range=", s.image_curve.min(), s.image_curve.max(),
            )
            c = np.asarray(s.dense_curve, dtype=np.float64)
            if c.ndim == 2 and c.shape[1] == 2 and len(c) >= max(2 * args.half_width + 1, args.patch_size):
                all_curve_items.append((cf, i, s))

    if args.max_curves is not None and len(all_curve_items) > args.max_curves:
        idxs = rng.choice(len(all_curve_items), size=args.max_curves, replace=False)
        all_curve_items = [all_curve_items[int(i)] for i in idxs]

    # shuffle curve order so visuals are spread across different images
    rng.shuffle(all_curve_items)

    print(f"[INFO] evaluating {len(all_curve_items)} curves from {args.curve_dir}")

    all_rows: list[PatchRecord] = []
    error_buckets: dict[str, list[float]] = {name: [] for name in models}
    rotated_error_buckets: dict[str, list[float]] = {name: [] for name in models}

    visual_count = 0
    visual_images_seen: set[str] = set()
    image_dir = Path(args.image_dir)

    for curve_file, curve_idx, sample in all_curve_items:
        dense_curve = np.asarray(sample.dense_curve, dtype=np.float64)
        image_curve = np.asarray(sample.image_curve, dtype=np.float64)
        closed = bool(sample.closed)

        image_path = match_image_path(image_dir, curve_file)
        image = plt.imread(image_path) if image_path is not None and image_path.exists() else None

        n = len(dense_curve)
        if closed:
            valid_centers = np.arange(0, n)
        else:
            valid_centers = np.arange(args.half_width, n - args.half_width)
        if len(valid_centers) == 0:
            continue

        sample_centers = rng.choice(valid_centers, size=min(args.patches_per_curve, len(valid_centers)), replace=False)

        for center_idx in sample_centers:
            patch = sample_patch_around_index(
                curve_points=dense_curve,
                center_index=int(center_idx),
                patch_size=args.patch_size,
                half_width=args.half_width,
                mode="jittered_symmetric",
                closed=closed,
                rng=rng,
                jitter_fraction=0.45,
            )

            gt = fd_tangent(dense_curve, int(center_idx), closed=closed)
            gt = normalize(gt)

            R = rotation_matrix(args.transform_angle_deg)
            transformed_patch_centered = (R @ patch.centered_points.T).T
            transformed_gt = normalize(R @ gt)

            for model_name, model in models.items():
                pred, weights = predict(model, patch.centered_points, device)
                pred = normalize(pred)

                pred_rot, _ = predict(model, transformed_patch_centered, device)
                pred_rot = normalize(pred_rot)

                cosine = float(np.clip(np.dot(pred, gt), -1.0, 1.0))
                err = angle_deg(pred, gt, sign_invariant=True)
                rot_err = angle_deg(pred_rot, transformed_gt, sign_invariant=True)

                error_buckets[model_name].append(err)
                rotated_error_buckets[model_name].append(rot_err)

                all_rows.append(
                    PatchRecord(
                        model_name=model_name,
                        image_file=image_path.name if image_path is not None else curve_file.name,
                        curve_file=curve_file.name,
                        curve_index=int(curve_idx),
                        patch_center_index=int(center_idx),
                        angular_error_deg=err,
                        cosine=cosine,
                        pred_norm=float(np.linalg.norm(pred)),
                        gt_norm=float(np.linalg.norm(gt)),
                        rotated_angular_error_deg=rot_err,
                        transform_angle_deg=float(args.transform_angle_deg),
                    )
                )

                # spread visuals across images first, then fill remainder
                should_save = False
                image_tag = image_path.name if image_path is not None else curve_file.name
                if visual_count < args.save_num_visuals:
                    if image_tag not in visual_images_seen:
                        should_save = True
                        visual_images_seen.add(image_tag)
                    elif len(visual_images_seen) >= min(args.save_num_visuals, len(curve_files)):
                        should_save = True

                if should_save:
                    out_path = vis_dir / f"{model_name}_{curve_file.stem}_c{curve_idx}_p{int(center_idx)}.png"
                    save_overlay_figure(
                        image=image,
                        image_curve=image_curve,
                        dense_curve=dense_curve,
                        patch_points_dense=patch.points,
                        center_idx=int(center_idx),
                        closed=closed,
                        pred_vec=pred,
                        gt_vec=gt,
                        weights=weights,
                        transformed_patch=transformed_patch_centered,
                        transformed_pred=pred_rot,
                        transformed_gt=transformed_gt,
                        transform_angle_deg=float(args.transform_angle_deg),
                        title=(
                            f"{model_name} | {curve_file.name} | curve {curve_idx} | "
                            f"center {int(center_idx)} | err={err:.2f}° | rot_err={rot_err:.2f}°"
                        ),
                        out_path=out_path,
                    )
                    visual_count += 1

    csv_path = out_root / "patch_metrics.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = list(asdict(all_rows[0]).keys()) if all_rows else list(PatchRecord.__annotations__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))

    summary = {
        name: {
            "base": angular_error_stats(error_buckets[name]),
            "rotated_patch": angular_error_stats(rotated_error_buckets[name]),
        }
        for name in models
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    plt.figure(figsize=(9, 5))
    plotted = False
    for name, errs in error_buckets.items():
        if errs:
            plt.hist(errs, bins=30, alpha=0.45, label=f"{name} base")
            plotted = True
    if plotted:
        plt.legend()
    plt.xlabel("angular error (deg, sign-invariant)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_root / "error_histograms.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plotted = False
    for name, errs in rotated_error_buckets.items():
        if errs:
            plt.hist(errs, bins=30, alpha=0.45, label=f"{name} rotated")
            plotted = True
    if plotted:
        plt.legend()
    plt.xlabel("rotated-patch angular error (deg, sign-invariant)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_root / "rotated_error_histograms.png", dpi=160)
    plt.close()

    print("\n[INFO] Summary:")
    print(json.dumps(summary, indent=2))
    print(f"[INFO] patch metrics -> {csv_path}")
    print(f"[INFO] summary -> {out_root / 'summary.json'}")
    print(f"[INFO] visuals -> {vis_dir}")


if __name__ == "__main__":
    main()
