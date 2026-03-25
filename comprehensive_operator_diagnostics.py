from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Ensure project root is in the path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT_CANDIDATES = [THIS_FILE.parent, THIS_FILE.parent.parent]
for candidate in PROJECT_ROOT_CANDIDATES:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from datasets.tangent_dataset import TangentDataset
from models.tangent_model import TangentOperatorModel
from evaluate_operator_diagnostics_fixed import (
    reproduce_sample, 
    predict_patch, 
    cosine_and_angle, 
    patch_spacing_metrics
)

def parse_int_list(val) -> list[int]:
    """Helper to parse MLP dims from config which might be strings or lists."""
    if isinstance(val, list):
        return val
    return [int(x.strip()) for x in str(val).split(",") if x.strip()]

def analyze_operator_spectral_properties(model, patch_size):
    """Checks if W satisfies the fundamental null-space of derivatives."""
    model.eval()
    device = next(model.parameters()).device
    
    # Sample weights for a flat line to see 'base' stencil behavior
    flat_patch = torch.linspace(-1, 1, patch_size).view(1, patch_size, 1).repeat(1, 1, 2)
    with torch.no_grad():
        out = model(flat_patch.to(device))
        w = out["weights"][0].cpu().numpy()

    # C1: Translation Invariance (Sum of weights must be 0)
    dc_gain = np.sum(w)
    
    # C2: Stencil Symmetry (First derivatives should be anti-symmetric: w[i] ≈ -w[-(i+1)])
    symmetry_err = np.linalg.norm(w + w[::-1])
    
    # C3: Second Moment
    indices = np.arange(patch_size) - (patch_size // 2)
    curv_potential = np.sum(w * (indices**2))
    
    return {
        "dc_gain_drift": float(dc_gain),
        "anti_symmetry_error": float(symmetry_err),
        "curvature_potential": float(curv_potential),
        "weight_vec": w.tolist()
    }

def main():
    parser = argparse.ArgumentParser(description="Headless deep diagnostics for Tangent Operator")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the training run directory")
    parser.add_argument("--test-curve-dir", type=str, required=True, help="Path to precomputed test curves")
    parser.add_argument("--test-length", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Fallback arguments in case run_config.json is missing
    parser.add_argument("--patch-size", type=int, default=5, help="Fallback patch size")
    parser.add_argument("--half-width", type=int, default=12, help="Fallback half width")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    diag_dir = run_dir / "deep_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    
    # 1. Load configuration automatically or use fallbacks
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        print(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        patch_size = config.get("patch_size", args.patch_size)
        half_width = config.get("half_width", args.half_width)
        point_mlp_dims = parse_int_list(config.get("point_mlp_dims", "64,64,128"))
        head_dims = parse_int_list(config.get("head_dims", "128,64"))
        use_batchnorm = config.get("use_batchnorm", True)
    else:
        print(f"Warning: {config_path} not found. Using fallback arguments (patch_size={args.patch_size}).")
        patch_size = args.patch_size
        half_width = args.half_width
        point_mlp_dims = [64, 64, 128]
        head_dims = [128, 64]
        use_batchnorm = True

    # 2. Initialize Model with correct architecture
    model = TangentOperatorModel(
        patch_size=patch_size,
        point_mlp_dims=point_mlp_dims,
        head_dims=head_dims,
        use_batchnorm=use_batchnorm
    )
    
    checkpoint_path = run_dir / "checkpoints/best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device).eval()

    # 3. Initialize Dataset with matching patch dimensions
    dataset = TangentDataset(
        length=args.test_length, 
        family="euclidean", 
        num_curve_points=1000, 
        patch_size=patch_size, 
        half_width=half_width, 
        use_precomputed_curves=True,
        precomputed_curve_dir=args.test_curve_dir, 
        seed=123+20000
    )

    results = []
    print(f"Evaluating {len(dataset)} samples for deep diagnostics...")
    
    for i in range(len(dataset)):
        s = reproduce_sample(dataset, i)
        p1, p2 = predict_patch(model, s["anchor_patch"], device)
        gt1, gt2 = s["gt_first"], s["gt_second"]
        
        n2_gt = np.linalg.norm(gt2)
        n2_pred = np.linalg.norm(p2)
        
        # Metric 1: Relative Error for Curvature
        rel_err_2 = float(np.linalg.norm(p2 - gt2) / (n2_gt + 1e-6))
        
        # Metric 2: Orthogonality Leakage (Tangent vs Curvature should be orthogonal)
        leakage = float(np.abs(np.dot(p1, p2)) / (np.linalg.norm(p1) * n2_pred + 1e-8))
        
        spacing = patch_spacing_metrics(s["anchor_patch"])
        results.append({
            "gt_k": float(n2_gt), 
            "pred_k": float(n2_pred), 
            "rel_err": rel_err_2,
            "angle_err": float(cosine_and_angle(p2, gt2)[1]),
            "leakage": leakage, 
            "spacing_cv": float(spacing["spacing_cv"])
        })

    df = pd.DataFrame(results)
    
    # 4. Save JSON Statistics
    summary = {
        "spectral_props": analyze_operator_spectral_properties(model, patch_size),
        "correlations": {
            "curvature_vs_error": float(df['gt_k'].corr(df['rel_err'])) if not df['gt_k'].isna().all() else 0.0,
            "spacing_jitter_vs_error": float(df['spacing_cv'].corr(df['rel_err'])) if not df['spacing_cv'].isna().all() else 0.0
        },
        "error_thresholds": {
            "pct_samples_rel_err_above_1": float((df['rel_err'] > 1.0).mean())
        }
    }
    
    with open(diag_dir / "stats.json", "w") as f:
        json.dump(summary, f, indent=4)

    # 5. Generate and save plots headlessly
    plt.ioff()
    
    # Plot A: SNR Analysis
    fig_snr, ax_snr = plt.subplots(figsize=(8, 6))
    ax_snr.scatter(df['gt_k'], df['rel_err'], alpha=0.2)
    ax_snr.axhline(1.0, color='r', linestyle='--', label='Noise Floor (Error = Signal)')
    ax_snr.set_xscale('log')
    ax_snr.set_yscale('log')
    ax_snr.set_xlabel('Ground Truth Curvature Magnitude')
    ax_snr.set_ylabel('Relative Error')
    ax_snr.set_title('Curvature SNR Breakdown')
    ax_snr.legend()
    fig_snr.savefig(diag_dir / "snr_analysis.png", dpi=180, bbox_inches="tight")
    plt.close(fig_snr)

    # Plot B: Spectral Leakage
    fig_leak, ax_leak = plt.subplots(figsize=(8, 6))
    ax_leak.hist(df['leakage'].dropna(), bins=50, color='tab:orange', alpha=0.7)
    ax_leak.set_xlabel('|cos(Predicted Tangent, Predicted Curvature)|')
    ax_leak.set_ylabel('Count')
    ax_leak.set_title('Orthogonality Leakage (Should be near 0)')
    fig_leak.savefig(diag_dir / "spectral_leakage.png", dpi=180, bbox_inches="tight")
    plt.close(fig_leak)
    
    # Plot C: Norm Regression
    fig_norm, ax_norm = plt.subplots(figsize=(8, 6))
    ax_norm.scatter(df['gt_k'], df['pred_k'], alpha=0.2, color='tab:green')
    max_val = max(df['gt_k'].max(), df['pred_k'].max())
    ax_norm.plot([0, max_val], [0, max_val], 'r--', label='Ideal')
    ax_norm.set_xlabel('Ground Truth Curvature Magnitude')
    ax_norm.set_ylabel('Predicted Curvature Magnitude')
    ax_norm.set_title('Curvature Magnitude Regression')
    ax_norm.legend()
    fig_norm.savefig(diag_dir / "norm_regression.png", dpi=180, bbox_inches="tight")
    plt.close(fig_norm)

    # 6. Save raw metrics to CSV
    df.to_csv(diag_dir / "deep_metrics.csv", index=False)

    print(f"✅ Deep diagnostics completed successfully.")
    print(f"Results logged to: {diag_dir}")

if __name__ == "__main__":
    main()
