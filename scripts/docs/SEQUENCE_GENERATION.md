# Sequence Generation (Baseline vs StycoGAN)

This repository generates qualitative sequences by producing a fixed number of frames per identity seed.

## Outputs
- Baseline sequences: Drive `/experiments/outputs/stylegan_sequences/seed_X/`
- StycoGAN sequences: Drive `/experiments/outputs/stycogan_sequences/seed_X/`

## Notes
- For reproducibility, each frame uses a deterministic seed mapping: `seed*1000 + t`.
- Temporal window default: `T = 8` frames.
- Update `STYCO_PKL` to point to the fine-tuned StycoGAN snapshot.
