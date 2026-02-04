# Reproducibility

This document describes the minimal requirements and procedures to reproduce the StycoGAN experiments reported in this repository. The focus is on ensuring that all experimental results, particularly sequence generation in Phase B, can be independently verified.

## Environment

The experiments were conducted using a StyleGAN2-ADA (PyTorch) codebase extended with StycoGAN components.

Minimum requirements:
- Python 3.8+
- PyTorch (CUDA-enabled recommended)
- NVIDIA GPU (tested on single-GPU setup)
- StyleGAN2-ADA (PyTorch) repository available in the environment

The code assumes that `legacy.py` from StyleGAN2-ADA is importable. If required, the following environment variable should be set:

```bash
export PYTHONPATH=/path/to/stylegan2-ada-pytorch:$PYTHONPATH

## Checkpoints

Two types of generator checkpoints are used:

Baseline checkpoint: a frame-independent generator trained without temporal regularization.

StycoGAN checkpoint: a generator trained with temporal consistency regularization.

Checkpoint files are provided externally and are not included in this repository.

## Experimental Phases

### Phase A: Model Training (Summary)

Phase A refers to the training or fine-tuning of generator models. Training scripts are provided in the scripts/ directory. Due to computational cost, training is not expected to be reproduced in full for verification of Phase B results.

### Phase B: Sequence Generation


Phase B evaluates temporal consistency by generating a sequence of frames from a fixed identity latent code.

Reproducibility in Phase B is ensured by:
- fixing the random seed
- fixing the identity latent code
- using the same number of frames \(T\)
- using identical truncation and noise settings for both baseline and StycoGAN


The complete procedure for Phase B is described in docs/SEQUENCE_GENERATION.md.

Determinism and Variability

While GPU-based generation may introduce minor numerical variability, the following measures ensure meaningful comparison:

identical seeds are used for baseline and StycoGAN

all synthesis parameters are held constant across models

stochastic noise is controlled and documented

Observed differences between baseline and StycoGAN outputs therefore reflect differences in temporal modeling rather than random variation.

Recorded Metadata

Each experimental run automatically saves a `meta.json` file containing:
- random seed
- number of frames
- truncation value
- noise mode
- generator checkpoint identifier


This metadata allows exact reconstruction of experimental conditions.

Notes for Reviewers

The repository is structured to separate:

scripts (scripts/)

documentation (docs/)

configuration descriptions (configs/)

All figure-ready outputs used for qualitative evaluation are generated directly by the scripts provided, without manual post-processing.
