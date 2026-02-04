#!/usr/bin/env bash
# Generate sequences for baseline vs StycoGAN using StyleGAN2-ADA generate.py
# Output: a folder of frames per seed range (for temporal comparison)

set -e

# === PATHS (Colab) ===
OUT_ROOT=/content/drive/MyDrive/experiments/outputs

# Baseline snapshot (StyleGAN)
BASE_PKL=/content/drive/MyDrive/experiments/styco_runs_fixed/00000-celeba_hq_256-stylegan2-kimg20-batch16-noaug/network-snapshot-000020.pkl

# StycoGAN snapshot (replace with your actual fine-tuned snapshot path)
STYCO_PKL=/content/drive/MyDrive/experiments/styco_runs_fixed/REPLACE_WITH_STYCOGAN_SNAPSHOT.pkl

# Repo folder
STYLEGAN_DIR=/content/stylegan2-ada-pytorch

# === SEQUENCE SETTINGS ===
SEEDS_START=0
SEEDS_END=100        # inclusive range style like 0-100
TRUNC=1
RES=256
N_FRAMES=8           # temporal window you want to visualize (T)

echo "OUT_ROOT   : ${OUT_ROOT}"
echo "BASE_PKL   : ${BASE_PKL}"
echo "STYCO_PKL  : ${STYCO_PKL}"
echo "Seeds      : ${SEEDS_START}-${SEEDS_END}"
echo "Frames (T) : ${N_FRAMES}"

mkdir -p "${OUT_ROOT}/stylegan_sequences"
mkdir -p "${OUT_ROOT}/stycogan_sequences"

cd "${STYLEGAN_DIR}"

# --- Baseline sequences ---
echo "[1/2] Generating BASELINE sequences..."
for seed in $(seq ${SEEDS_START} ${SEEDS_END}); do
  outdir="${OUT_ROOT}/stylegan_sequences/seed_${seed}"
  mkdir -p "${outdir}"

  # Generate N_FRAMES images for the same seed by varying class of "frame index"
  # Here we encode "frame index" into the random seed deterministically: seed*1000 + t
  for t in $(seq 0 $((N_FRAMES-1))); do
    python generate.py --outdir="${outdir}" --trunc="${TRUNC}" --seeds=$((seed*1000 + t)) --network="${BASE_PKL}"
  done
done

# --- StycoGAN sequences ---
echo "[2/2] Generating STYCOGAN sequences..."
for seed in $(seq ${SEEDS_START} ${SEEDS_END}); do
  outdir="${OUT_ROOT}/stycogan_sequences/seed_${seed}"
  mkdir -p "${outdir}"

  for t in $(seq 0 $((N_FRAMES-1))); do
    python generate.py --outdir="${outdir}" --trunc="${TRUNC}" --seeds=$((seed*1000 + t)) --network="${STYCO_PKL}"
  done
done

echo "Done. Outputs saved under:"
echo " - ${OUT_ROOT}/stylegan_sequences/"
echo " - ${OUT_ROOT}/stycogan_sequences/"
