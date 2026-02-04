#!/usr/bin/env python3
"""
generate_xt_sequence.py

Phase B: Generate Xt sequence (T frames) from a fixed identity latent (z fixed),
and optionally compare baseline vs StycoGAN by running this script twice with
different network.pkl.

Outputs (in outdir):
- frames/frame_000.png ... frame_{T-1}.png
- strip.png : 1-row contact sheet (figure-ready)
- grid.png  : grid contact sheet
- meta.json : run metadata for reproducibility

Assumptions:
- You run this inside a StyleGAN2-ADA(-PyTorch) / StycoGAN environment where
  `legacy.py` is importable. If not, set PYTHONPATH to the repo that has legacy.py.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Import legacy loader from StyleGAN2-ADA PyTorch
try:
    import legacy  # type: ignore
except Exception as e:
    raise SystemExit(
        "ERROR: Cannot import 'legacy'.\n"
        "Fix: run inside stylegan2-ada-pytorch environment OR set PYTHONPATH, e.g.:\n"
        "  export PYTHONPATH=/path/to/stylegan2-ada-pytorch:$PYTHONPATH\n"
        f"Original error: {e}"
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_uint8_image(img: torch.Tensor) -> Image.Image:
    """
    img: torch tensor [C,H,W] in [-1,1]
    """
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    arr = img.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def make_grid(images, cols: int, pad: int = 8, bg: int = 255) -> Image.Image:
    if not images:
        raise ValueError("No images to grid.")
    w, h = images[0].size
    rows = math.ceil(len(images) / cols)

    grid_w = cols * w + (cols + 1) * pad
    grid_h = rows * h + (rows + 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), (bg, bg, bg))

    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        canvas.paste(im, (x, y))
    return canvas


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, help="Path to network .pkl (baseline or StycoGAN)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--T", type=int, default=8, help="Number of frames")
    ap.add_argument("--seed", type=int, default=42, help="Seed for identity z and schedule")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")

    # Key controls for "fixed identity latent"
    ap.add_argument(
        "--z_mode",
        default="fixed",
        choices=["fixed", "random_walk", "lerp"],
        help="How z_t is formed across time. Use 'fixed' for Phase B fixed identity.",
    )
    ap.add_argument("--rw_sigma", type=float, default=0.04, help="Random-walk step size (if z_mode=random_walk)")
    ap.add_argument("--lerp_seed2", type=int, default=123, help="Second seed for lerp endpoint (if z_mode=lerp)")

    # Synthesis controls
    ap.add_argument("--trunc", type=float, default=1.0, help="Truncation psi")
    ap.add_argument(
        "--noise_mode",
        default="random",
        choices=["random", "const", "none"],
        help=(
            "Noise mode passed into G.synthesis if supported.\n"
            "Recommended for visual flicker test: fixed z + noise_mode=random.\n"
            "For fully deterministic frames: noise_mode=const."
        ),
    )

    # Figure settings
    ap.add_argument("--grid_cols", type=int, default=8, help="Columns for grid.png (default: 8)")
    ap.add_argument("--pad", type=int, default=8, help="Padding in pixels for strip/grid sheets")
    ap.add_argument("--label", default="", help="Optional label stored in meta.json")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    seed_everything(args.seed)

    # Load network
    with open(args.network, "rb") as f:
        net = legacy.load_network_pkl(f)

    G = net["G_ema"].to(device).eval()

    # Read model dims
    z_dim = int(getattr(G, "z_dim", 512))
    c_dim = int(getattr(G, "c_dim", 0))

    # Conditioning
    c = torch.zeros([1, c_dim], device=device) if c_dim > 0 else None

    # Build z schedule
    rng = np.random.RandomState(args.seed)
    z0 = rng.randn(1, z_dim).astype(np.float32)

    Z = []
    if args.z_mode == "fixed":
        Z = [z0 for _ in range(args.T)]
    elif args.z_mode == "random_walk":
        cur = z0.copy()
        for _ in range(args.T):
            cur = cur + rng.randn(1, z_dim).astype(np.float32) * args.rw_sigma
            Z.append(cur.copy())
    else:  # lerp
        rng2 = np.random.RandomState(args.lerp_seed2)
        z1 = rng2.randn(1, z_dim).astype(np.float32)
        for t in range(args.T):
            a = 0.0 if args.T == 1 else (t / (args.T - 1))
            Z.append(((1 - a) * z0 + a * z1).astype(np.float32))

    # Generate frames
    images = []
    for t in range(args.T):
        # Make stochastic components reproducible per-frame
        seed_everything(args.seed + t * 1000)

        z = torch.from_numpy(Z[t]).to(device)

        # Call generator (StyleGAN2-ADA style signature)
        if c is None:
            img = G(z, None, truncation_psi=args.trunc, noise_mode=args.noise_mode)
        else:
            img = G(z, c, truncation_psi=args.trunc, noise_mode=args.noise_mode)

        pil = to_uint8_image(img[0])
        pil.save(frames_dir / f"frame_{t:03d}.png")
        images.append(pil)

    # Figure-ready outputs
    strip = make_grid(images, cols=args.T, pad=args.pad, bg=255)
    strip.save(outdir / "strip.png")

    grid_cols = max(1, int(args.grid_cols))
    grid = make_grid(images, cols=grid_cols, pad=args.pad, bg=255)
    grid.save(outdir / "grid.png")

    meta = {
        "network": str(args.network),
        "outdir": str(outdir),
        "T": int(args.T),
        "seed": int(args.seed),
        "z_mode": args.z_mode,
        "rw_sigma": float(args.rw_sigma),
        "lerp_seed2": int(args.lerp_seed2),
        "trunc": float(args.trunc),
        "noise_mode": args.noise_mode,
        "z_dim": z_dim,
        "c_dim": c_dim,
        "label": args.label,
        "frames_dir": str(frames_dir),
    }
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] frames: {frames_dir}")
    print(f"[OK] strip : {outdir/'strip.png'}")
    print(f"[OK] grid  : {outdir/'grid.png'}")
    print(f"[OK] meta  : {outdir/'meta.json'}")


if __name__ == "__main__":
    main()


A) Baseline (StyleGAN2-ADA) — fixed z, T=8
python scripts/generate_xt_sequence.py \
  --network /path/to/baseline.pkl \
  --outdir outputs/phaseB/baseline_seed42 \
  --T 8 --seed 42 \
  --z_mode fixed \
  --noise_mode random \
  --trunc 1.0

B) StycoGAN — fixed z, T=8
python scripts/generate_xt_sequence.py \
  --network /path/to/stycogan.pkl \
  --outdir outputs/phaseB/stycogan_seed42 \
  --T 8 --seed 42 \
  --z_mode fixed \
  --noise_mode random \
  --trunc 1.0
