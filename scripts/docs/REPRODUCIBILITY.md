# Reproducibility Notes

## Dataset
- CelebA-HQ (256×256)
- Dataset ZIP prepared using StyleGAN2-ADA dataset_tool.py
- Large artifacts (raw data, ZIP, checkpoints) are stored externally (e.g., Drive)

## Baseline
- StyleGAN2-ADA baseline: kimg=20, batch=16, no augmentation
- Baseline snapshot is used as the resume checkpoint for StycoGAN fine-tuning

## StycoGAN Fine-tuning
- lambda_styco = 0.3
- temporal window T = 8

## Environment
- Platform: Google Colab
- GPU: NVIDIA L4
- Framework: PyTorch + StyleGAN2-ADA
