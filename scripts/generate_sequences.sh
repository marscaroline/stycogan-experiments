#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Phase B: Sequence Generation (Xt)
# - Fixed identity latent (z fixed)
# - Generate T frames (default T=8)
# - Compare Baseline vs StycoGAN using the SAME seed
#
# Outputs:
#   outputs/phaseB/<run_name>/
#     ├── baseline/strip.png, grid.png, frames/
#     ├── stycogan/strip.png, grid.png, frames/
#     └── figure_baseline_vs_stycogan.png (optional)
# ============================================================

# ---- Required ----
BASELINE_PKL="${BASELINE_PKL:-}"
STYCOGAN_PKL="${STYCOGAN_PKL:-}"

# ---- Optional (defaults) ----
OUTROOT="${OUTROOT:-outputs/phaseB}"
RUN_NAME="${RUN_NAME:-seed42_T8_fixedz}"
T="${T:-8}"
SEED="${SEED:-42}"
TRUNC="${TRUNC:-1.0}"

# For Phase B visual comparison:
# - fixed z + noise_mode=random often reveals baseline flicker more clearly.
NOISE_MODE="${NOISE_MODE:-random}"       # random | const | none
Z_MODE="${Z_MODE:-fixed}"               # fixed | random_walk | lerp
GRID_COLS="${GRID_COLS:-8}"
PAD="${PAD:-8}"

# Optional: create combined figure (requires PIL)
MAKE_COMBINED="${MAKE_COMBINED:-1}"     # 1=yes, 0=no

usage() {
  cat <<EOF
Usage:
  BASELINE_PKL=/path/baseline.pkl STYCOGAN_PKL=/path/stycogan.pkl \\
  bash scripts/generate_sequences.sh

Optional env vars:
  OUTROOT=outputs/phaseB
  RUN_NAME=seed42_T8_fixedz
  T=8
  SEED=42
  TRUNC=1.0
  NOISE_MODE=random        (random|const|none)
  Z_MODE=fixed             (fixed|random_walk|lerp)
  GRID_COLS=8
  PAD=8
  MAKE_COMBINED=1          (1|0)
EOF
}

if [[ -z "$BASELINE_PKL" || -z "$STYCOGAN_PKL" ]]; then
  echo "ERROR: BASELINE_PKL and STYCOGAN_PKL must be set."
  usage
  exit 1
fi

OUTDIR="${OUTROOT}/${RUN_NAME}"
BASE_OUT="${OUTDIR}/baseline"
STY_OUT="${OUTDIR}/stycogan"

mkdir -p "$OUTDIR"

echo "============================================================"
echo "[Phase B] Sequence Generation (Xt)"
echo "OUTDIR     : $OUTDIR"
echo "T / SEED   : $T / $SEED"
echo "Z_MODE     : $Z_MODE"
echo "NOISE_MODE : $NOISE_MODE"
echo "TRUNC      : $TRUNC"
echo "============================================================"

echo "[1/2] Baseline..."
python scripts/generate_xt_sequence.py \
  --network "$BASELINE_PKL" \
  --outdir "$BASE_OUT" \
  --T "$T" --seed "$SEED" \
  --z_mode "$Z_MODE" \
  --noise_mode "$NOISE_MODE" \
  --trunc "$TRUNC" \
  --grid_cols "$GRID_COLS" \
  --pad "$PAD" \
  --label "baseline"

echo "[2/2] StycoGAN..."
python scripts/generate_xt_sequence.py \
  --network "$STYCOGAN_PKL" \
  --outdir "$STY_OUT" \
  --T "$T" --seed "$SEED" \
  --z_mode "$Z_MODE" \
  --noise_mode "$NOISE_MODE" \
  --trunc "$TRUNC" \
  --grid_cols "$GRID_COLS" \
  --pad "$PAD" \
  --label "stycogan"

if [[ "$MAKE_COMBINED" == "1" ]]; then
  echo "[3/3] Combined figure..."
  python - <<'PY'
from pathlib import Path
from PIL import Image
import os

outdir = Path(os.environ["OUTDIR"])
b = Image.open(outdir / "baseline" / "strip.png").convert("RGB")
s = Image.open(outdir / "stycogan" / "strip.png").convert("RGB")

pad = 16
W = max(b.size[0], s.size[0]) + pad*2
H = b.size[1] + s.size[1] + pad*3

canvas = Image.new("RGB", (W, H), (255,255,255))
canvas.paste(b, (pad, pad))
canvas.paste(s, (pad, pad*2 + b.size[1]))

out = outdir / "figure_baseline_vs_stycogan.png"
out.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out)
print(f"[OK] saved: {out}")
PY
fi

echo "✅ Done."
echo "Baseline strip : $BASE_OUT/strip.png"
echo "StycoGAN strip : $STY_OUT/strip.png"
if [[ "$MAKE_COMBINED" == "1" ]]; then
  echo "Combined figure: $OUTDIR/figure_baseline_vs_stycogan.png"
fi
