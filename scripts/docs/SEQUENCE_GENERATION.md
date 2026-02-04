# Sequence Generation (Phase B)

This document describes Phase B of the StycoGAN experiments: generating a temporally ordered image sequence \(X_t\) from a fixed identity latent code and producing figure-ready visual comparisons between a baseline generator and StycoGAN.

## Objective

The objective of Phase B is to evaluate temporal consistency under a fixed identity condition. Given a single latent code \(z\), the generator is used to produce a sequence of \(T\) frames (\(T = 8\)):

- **Baseline generator**: operates in a frame-independent manner and may exhibit temporal flickering.
- **StycoGAN**: integrates temporal regularization and is expected to generate perceptually stable sequences across time.

In all Phase B experiments, the identity latent is fixed: \( z_t = z \) for all \( t \).


## Scripts

Phase B relies on the following scripts:

- `scripts/generate_xt_sequence.py`  
  Generates a sequence of frames for a given generator checkpoint and saves figure-ready outputs.

- `scripts/generate_sequences.sh`  
  Wrapper script that runs sequence generation for both baseline and StycoGAN using identical settings.

## Experimental Procedure

Sequence generation is performed using the same configuration for both baseline and StycoGAN models to ensure a fair comparison.

1. A fixed latent code \(z\) is sampled using a predefined random seed.
2. The generator produces \(T = 8\) frames sequentially.
3. Stochastic synthesis noise is enabled to expose potential temporal instability in frame-independent generators.
4. The same seed and parameters are reused for both models.

## Execution

Run Phase B using the wrapper script:

```bash
BASELINE_PKL=/path/to/baseline.pkl \
STYCOGAN_PKL=/path/to/stycogan.pkl \
bash scripts/generate_sequences.sh

All parameters required for Phase B (number of frames, seed, truncation, and noise mode) are defined inside the script to ensure consistency and reproducibility.

Outputs

Each run produces the following directory structure:
outputs/phaseB/<run_name>/
├── baseline/
│   ├── frames/
│   ├── strip.png
│   └── meta.json
├── stycogan/
│   ├── frames/
│   ├── strip.png
│   └── meta.json
└── figure_baseline_vs_stycogan.png
`strip.png` contains all \(T\) frames arranged in temporal order and is directly usable as a figure.


figure_baseline_vs_stycogan.png stacks baseline and StycoGAN results for direct visual comparison.

Reproducibility

Each sequence generation run saves a `meta.json` file containing:
- random seed
- number of frames \(T\)
- truncation value
- noise mode
- generator checkpoint used


Using the same configuration guarantees that differences between baseline and StycoGAN outputs reflect temporal modeling effects rather than identity variation.
