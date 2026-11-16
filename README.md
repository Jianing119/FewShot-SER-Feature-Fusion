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
- [x] Dataset preprocessing and MFCC extraction
- [x] MFCC + RBF SVM validation baseline
- [ ] Few-shot experiments
- [ ] Feature fusion experiments
- [ ] Results and analysis

## Reproducibility

### Day 4: MFCC + RBF SVM baseline

From the project root in PowerShell, after generating the Day 3 feature archive:

```powershell
.\.venv\Scripts\python.exe scripts/train_svm.py
```

The input is `features/mfcc_features.npz` (768 × 80). The training split
has 512 utterances from speakers 01–16; validation has 128 from speakers
17–20. The 128 utterances from speakers 21–24 remain reserved for test.
The fixed label set is `happy`, `sad`, `angry`, `fearful`.

Every candidate fits a fresh StandardScaler + RBF SVM pipeline on the training
split only. The predefined grid is C = {0.1, 1, 10, 100} and
gamma = {0.001, 0.01, 0.1}. Selection maximizes validation macro-F1, breaking
ties by higher UAR, higher accuracy, smaller C, then smaller gamma. UAR is
the mean recall across the four labels. Day 4 does not score or predict on
the test set or refit on validation.

- `results/mfcc_svm_val.csv`: accuracy, macro-F1 and UAR for all 12 validation
  candidates; `selected=1` marks the chosen row.
- `models/mfcc_svm.joblib`: selected pipeline fitted on training data only.
  This file is kept locally; load only trusted joblib artifacts.

These are validation results for model selection; they do not estimate final
test performance. This baseline uses all training utterances and precedes
the few-shot experiments.

The selected validation candidate was C=100, gamma=0.01: accuracy 0.6172, macro-F1 0.6217, and UAR 0.6172.

## References

- RAVDESS dataset: https://zenodo.org/records/1188976
- emotion2vec: https://huggingface.co/emotion2vec/emotion2vec_base

## Project Status

Work in progress — experimental results have not yet been finalized.
