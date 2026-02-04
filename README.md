# StycoGAN Experiments

This repository provides a **reproducible experimental pipeline** for **StycoGAN (Style–Temporal Coherence GAN)**.

The purpose of this repository is to document and support experimental evidence for:
- StyleGAN2-ADA baseline training
- StycoGAN fine-tuning with temporal coherence regularization
- Sequence-based qualitative evaluation

## Scope
This repository focuses on **experiment reproducibility**, not on hosting large datasets or trained model checkpoints.

## Included
- Experiment scripts and configurations
- Dataset preparation workflow (CelebA-HQ 256×256)
- StyleGAN2-ADA baseline setup
- StycoGAN fine-tuning scripts
- Documentation for reproducibility

## Not Included (by design)
- Raw datasets (CelebA-HQ images)
- Large dataset ZIP files
- Model checkpoints (.pkl)
- Large output sequences

These artifacts are stored externally (e.g., Google Drive) and referenced in configuration files.

## Environment
- Platform: Google Colab
- GPU: NVIDIA L4
- Framework: PyTorch, StyleGAN2-ADA

## License
MIT License
