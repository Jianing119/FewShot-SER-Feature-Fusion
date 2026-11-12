# Few-shot Speech Emotion Recognition with Feature Fusion

## Overview

This project investigates the effectiveness of traditional acoustic features and pre-trained speech representations for speech emotion recognition under limited labeled data conditions.

The study compares three approaches:

1. MFCC features with SVM
2. emotion2vec embeddings with SVM
3. Fusion of MFCC and emotion2vec features with SVM

## Research Questions

- How does the amount of labeled training data affect speech emotion recognition performance?
- Can pre-trained speech representations outperform traditional acoustic features in low-resource settings?
- Does feature fusion improve classification performance?

## Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song.

Four emotion categories will be considered: Happy, Sad, Angry, and Fearful.

Dataset: https://zenodo.org/records/1188976

The raw dataset is not included in this repository.

## Environment

- Python 3.10
- PyTorch 2.6.0
- CUDA 12.6
- FunASR
- emotion2vec_base
- Scikit-learn

## Current Progress

- [x] Python virtual environment setup
- [x] PyTorch CUDA verification
- [x] emotion2vec model initialization
- [x] Successful extraction of a 768-dimensional speech embedding
- [ ] Dataset preprocessing
- [ ] MFCC baseline
- [ ] Few-shot experiments
- [ ] Feature fusion experiments
- [ ] Results and analysis

## Reproducibility

Scripts, experimental configurations, and evaluation instructions will be added as the study progresses.

## References

- RAVDESS dataset: https://zenodo.org/records/1188976
- emotion2vec: https://huggingface.co/emotion2vec/emotion2vec_base

## Project Status

Work in progress — experimental results have not yet been finalized.
