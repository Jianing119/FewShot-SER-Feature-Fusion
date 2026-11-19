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
- [x] emotion2vec + RBF SVM validation baseline
- [x] MFCC + emotion2vec early-fusion validation
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

### Day 7 · MFCC + emotion2vec early fusion

After extracting both feature archives, run from the project root in PowerShell.
The first two commands generate the baseline CSVs required for the comparison.
Skip them if those CSVs are already available.

```powershell
.\.venv\Scripts\python.exe scripts/train_svm.py
.\.venv\Scripts\python.exe scripts/train_emotion2vec_svm.py
.\.venv\Scripts\python.exe scripts/train_fusion_svm.py
```

Both archives must have identical labels, split assignments, and file paths in
the same row order. The script checks the speaker sets from the filenames
(01–16 train, 17–20 validation, 21–24 test). It fits one StandardScaler per
branch using the 512 training utterances, then concatenates the 80 standardized
MFCC coordinates and 768 standardized emotion2vec coordinates in that order.
Each RBF SVM is fitted on the 512 fused training rows. The 128 validation rows
select the best configuration using the Day 4 grid and tie-breaking rule.
Test metadata is checked for split integrity. Test features are excluded from
transformation, fitting, hyperparameter selection, prediction, and scoring.

- `results/fusion_svm_val.csv` records all 12 validation candidates.
  `selected=1` marks the chosen configuration.
- `results/model_comparison_val.csv` combines the selected validation rows
  from all three models.
- `models/fusion_svm.joblib` stores the two fitted scalers in `scalers`,
  their concatenation order in `branch_order`, input widths in
  `branch_dimensions`, and the trained SVC in `svm`. Apply the stored scalers
  to matching raw features in that order, concatenate, and call `svm.predict`.
  The script reloads its own bundle and checks every validation prediction.

The selected fusion configuration (C=10, gamma=0.001) reached accuracy
0.703125, macro-F1 0.698886, and UAR 0.703125 on validation. Its macro-F1 was
2.27 percentage points above the emotion2vec baseline on this split.
The three scores per model are validation estimates from the full training
split. Keep the test set reserved for the final evaluation protocol.

Per-coordinate standardization still allows the wider branch to contribute
more to the squared Euclidean distances used by the RBF kernel. A follow-up
ablation can divide each standardized branch by the square root of its width
to compare approximately dimension-balanced distances, using the same
speaker splits and validation-only hyperparameter selection. The score
difference above measures the unweighted fusion setup on this validation split.

## References

- RAVDESS dataset: https://zenodo.org/records/1188976
- emotion2vec: https://huggingface.co/emotion2vec/emotion2vec_base
- FunASR emotion2vec implementation: https://github.com/modelscope/FunASR/blob/main/funasr/models/emotion2vec/model.py

## Project Status

Work in progress — experimental results have not yet been finalized.
