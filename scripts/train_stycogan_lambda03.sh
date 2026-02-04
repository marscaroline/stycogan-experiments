#!/usr/bin/env bash
# StycoGAN fine-tuning script (lambda = 0.3)
# Note: This script assumes you are running in Google Colab.

set -e

# === DATASET / OUTPUT PATHS ===
DATA_ZIP=/content/drive/MyDrive/celeba_hq_256.zip
OUTDIR=/content/drive/MyDrive/experiments/styco_runs_fixed

# Baseline snapshot (StyleGAN2-ADA baseline) used for StycoGAN fine-tuning
BASE_PKL=/content/drive/MyDrive/experiments/styco_runs_fixed/00000-celeba_hq_256-stylegan2-kimg20-batch16-noaug/network-snapshot-000020.pkl

# === TRAINING REPO PATH ===
STYLEGAN_DIR=/content/stylegan2-ada-pytorch

echo "DATA_ZIP:  ${DATA_ZIP}"
echo "OUTDIR:    ${OUTDIR}"
echo "BASE_PKL:  ${BASE_PKL}"
echo "lambda:    0.3"
echo "T:         8"

cd ${STYLEGAN_DIR}

python train.py \
  --outdir=${OUTDIR} \
  --data=${DATA_ZIP} \
  --gpus=1 \
  --cfg=stylegan2 \
  --batch=16 \
  --kimg=20 \
  --aug=noaug \
  --resume=${BASE_PKL} \
  --lambda_styco=0.3 \
  --styco_T=8
