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

### Day 5: emotion2vec feature extraction

From the project root in PowerShell, with the Day 1 CUDA environment and
the Day 2 WAV files and `data/metadata.csv` in place:

```powershell
$env:HF_HUB_DISABLE_XET = '1'
.\.venv\Scripts\python.exe scripts/extract_emotion2vec.py
.\.venv\Scripts\python.exe scripts/extract_emotion2vec.py --verify-only
```

The script reads the 768 WAVs in CSV order, checks their 48 kHz sample rate,
mixes channels to mono and resamples to 16 kHz before inference on `cuda:0`.
FunASR uses `emotion2vec/emotion2vec_base` from Hugging Face with
`granularity="utterance"` and `extract_embedding=True`.

`features/emotion2vec_features.npz` contains `X` (768 × 768, float32),
`labels`, `splits` and `file_paths` (768 entries each). Verification checks
the matrix for NaN and Inf and compares the three row fields to the CSV in
order. The split counts are 512 train, 128 validation and 128 test.
All rows are embedded independently with the pretrained model. This step
does not fit a classifier, select hyperparameters or score the test split.

Run one extraction process at a time. A checkpoint is written atomically
every 16 WAVs and on a handled error or Ctrl+C after new rows complete.
Rerun the same command to resume. If the CSV row order or values used for
alignment changed, the checkpoint is rejected. If you changed source audio
or the model environment, move the partial or final NPZ out of `features/`
before starting a fresh run. For a WAV error, the message gives the CSV line
number and file path. Once extraction completes, a regular rerun verifies the
existing archive. The raw audio, final archive and checkpoint are Git-ignored.

## References

- RAVDESS dataset: https://zenodo.org/records/1188976
- emotion2vec: https://huggingface.co/emotion2vec/emotion2vec_base
- FunASR emotion2vec implementation: https://github.com/modelscope/FunASR/blob/main/funasr/models/emotion2vec/model.py

## Project Status

Work in progress — experimental results have not yet been finalized.
