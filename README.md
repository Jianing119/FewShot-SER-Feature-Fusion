# Few-shot Speech Emotion Recognition with Feature Fusion

This project studies how labeled training data affects speech emotion recognition on unseen speakers. It compares three representations with the same RBF SVM classifier: **MFCC**, **pretrained emotion2vec**, and **MFCC + emotion2vec early fusion**. The experiments ask how performance scales with label budget, whether pretrained embeddings improve on acoustic features, and when their combination helps.

**Status:** Day 1–8 implementation and validation experiments are complete. The held-out test set is reserved for final evaluation.

## Data and features

The [RAVDESS dataset](https://zenodo.org/records/1188976) contributes **768 audio-only speech utterances** across four balanced classes: happy, sad, angry, and fearful. The three splits use disjoint speakers.

| Split | Actors | Utterances | Per class |
| --- | --- | ---: | ---: |
| Train | 01–16 | 512 | 128 |
| Validation | 17–20 | 128 | 32 |
| Held-out test | 21–24 | 128 | 32 |

| Representation | Construction | Shape / width |
| --- | --- | --- |
| MFCC | Mean and standard deviation of 40 MFCCs across frames at 16 kHz | 768 × 80 |
| emotion2vec | Frozen emotion2vec_base utterance embeddings | 768 × 768 |
| Early fusion | Standardize each branch using training rows, then concatenate MFCC followed by emotion2vec | 848 dimensions |

RAVDESS recordings are distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See the dataset record for attribution and commercial licensing. The pretrained checkpoint has separate terms in its [model card](https://huggingface.co/emotion2vec/emotion2vec_base).

## Validation results

### Full training set (512 utterances)

| RBF SVM features | C | Gamma | Accuracy | Macro-F1 | UAR |
| --- | ---: | ---: | ---: | ---: | ---: |
| MFCC | 100 | 0.01 | 0.617188 | 0.621694 | 0.617188 |
| emotion2vec | 10 | 0.001 | 0.679688 | 0.676197 | 0.679688 |
| Early fusion | 10 | 0.001 | **0.703125** | **0.69888633** | **0.703125** |

UAR is the mean per-class recall. See [model_comparison_val.csv](results/model_comparison_val.csv).

### Few-shot Macro-F1 on the fixed validation split

| Shots per class | Total train | MFCC | emotion2vec | Early fusion |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 20 | 34.41% | 41.11% | **41.71%** |
| 10 | 40 | 39.80% | **52.18%** | 51.95% |
| 20 | 80 | 44.49% | 56.50% | **57.61%** |
| 50 | 200 | 50.82% | 63.02% | **63.07%** |
| 128 (full) | 512 | 62.17% | 67.62% | **69.89%** |

Each 5–50-shot cell is the mean over five training subsamples. Each full-training cell represents one unique fit. On this split, emotion2vec improves over MFCC at all five budgets. Fusion leads at four budgets and is close to emotion2vec at 50-shot. See [fewshot_summary_val.csv](results/fewshot_summary_val.csv).

## Quick start and Day 1–8 progress

Run the commands from the repository root in Windows PowerShell.

### Day 1 · Environment and GPU smoke test

Use Python 3.10 and an NVIDIA driver compatible with the PyTorch 2.6.0 CUDA 12.6 build. [requirements-day1.txt](requirements-day1.txt) records the environment, including PyTorch, torchaudio, FunASR, librosa, and scikit-learn. The extra index supplies the CUDA wheels. [PyTorch installation reference](https://pytorch.org/get-started/previous-versions/)

```powershell
py -3.10 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-day1.txt
& $py -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
$env:HF_HUB_DISABLE_XET = '1'
& $py scripts/test_emotion2vec.py
```

The smoke test initializes the Hugging Face checkpoint through FunASR on CUDA and extracts a 768-dimensional embedding from the example WAV. The Xet setting works around the earlier Hugging Face download issue. Set it again in a new PowerShell session before downloading.

### Days 2–3 · Prepare RAVDESS and extract MFCCs

Download **`Audio_Speech_Actors_01-24.zip`** from the [RAVDESS release](https://zenodo.org/records/1188976) and extract the actor folders under `data/raw/RAVDESS/` so that, for example, `Actor_01/` is directly inside it.

```powershell
& $py scripts/prepare_dataset.py
& $py scripts/extract_mfcc.py
& $py scripts/check_mfcc.py
```

Day 2 generates [data/metadata.csv](data/metadata.csv) with paths, actor IDs, emotions, and split assignments. Day 3 writes and checks `features/mfcc_features.npz`.

### Days 4–7 · Train full-data baselines and extract emotion2vec

| Day | Completed work |
| --- | --- |
| 4 | Train the MFCC + RBF SVM baseline on all 512 training utterances. |
| 5 | Extract and verify all 768 frozen emotion2vec embeddings. |
| 6 | Train emotion2vec + RBF SVM on all 512 training utterances. |
| 7 | Check archive alignment and speaker assignments, then fit separate training-only scalers and train the 848-dimensional early-fusion RBF SVM. |

```powershell
& $py scripts/train_svm.py
$env:HF_HUB_DISABLE_XET = '1'
& $py scripts/extract_emotion2vec.py
& $py scripts/extract_emotion2vec.py --verify-only
& $py scripts/train_emotion2vec_svm.py
& $py scripts/train_fusion_svm.py
```

**Day 5 verification.** Extraction checks the source audio is at 48 kHz, converts to mono, and resamples to 16 kHz. FunASR uses `cuda:0` with `granularity="utterance"` and `extract_embedding=True`. The output, `features/emotion2vec_features.npz`, contains the float32 X matrix and labels, splits, and `file_paths` arrays. The verify-only command checks shape, finite values, and exact metadata alignment in CSV row order without running FunASR. A regular rerun also verifies an existing final archive.

Run one extractor at a time. It saves `features/emotion2vec_partial.npz` atomically every 16 WAVs and on handled errors or Ctrl+C after new rows complete. Rerun extraction to resume. A changed metadata order or alignment value invalidates the checkpoint. Move the partial and final archives aside before re-extracting after changes to the source audio or model environment. WAV errors identify the CSV line and file path.

**Days 4, 6, and 7 model selection.** Each candidate fits its standardization and classifier on training data. The grid uses C ∈ {0.1, 1, 10, 100} and gamma ∈ {0.001, 0.01, 0.1}. Rank by higher validation Macro-F1, UAR, and Accuracy, then by smaller C and gamma. The three candidate CSVs ([MFCC](results/mfcc_svm_val.csv), [emotion2vec](results/emotion2vec_svm_val.csv), [fusion](results/fusion_svm_val.csv)) contain split, C, gamma, accuracy, `macro_f1`, uar, and selected. The chosen row has `selected=1`. Day 7 reads both baseline CSVs to build the three-method comparison.

The selected joblib files are saved in models/. The fusion bundle contains scalers, `branch_order`, `branch_dimensions`, and svm. For inference, scale each raw branch with its saved scaler, concatenate in the stored order, then call `svm.predict`. Training checks that the reloaded model reproduces validation predictions. Load only trusted joblib artifacts.

### Day 8 · Run the few-shot experiment

With both feature archives present and aligned, run:

```powershell
& $py scripts/run_fewshot.py
# Optional: point to a different root containing features/ and results/
& $py scripts/run_fewshot.py --root "D:\SER-FewShot-Research"
```

For seeds **42, 123, 2026, 3407, 777**, the script permutes the training rows within each emotion, then takes the first **5, 10, 20, 50, or 128** without replacement. Budgets are nested within each seed, and all methods use identical sampled rows. Each scaler is fitted anew on the sampled training rows. Each method uses its full-training C and gamma from the results table at every budget. All runs score the same 128 validation utterances.

- [fewshot_runs_val.csv](results/fewshot_runs_val.csv) contains 75 records with split, method, `shots_per_class`, seed, `n_train`, C, gamma, `reused_full_train`, accuracy, `macro_f1`, and uar. At 128-shot the full training set is identical across seeds, so one fit per method supplies the first row and four rows marked `reused_full_train=1`.
- [fewshot_summary_val.csv](results/fewshot_summary_val.csv) contains 15 method-by-budget rows. It records split, method, `shots_per_class`, `n_train`, `n_records`, `n_fits`, and _mean/_std columns for accuracy, `macro_f1`, and uar. Standard deviations are sample estimates across the five subsamples at each smaller budget. At 128-shot, `n_records=5`, `n_fits=1`, and the standard deviations are blank.

## Interpretation and repository hygiene

The test audio is embedded independently and the test rows in the feature archives are checked for input integrity. Test data has not been used to fit scalers or classifiers, choose hyperparameters, or generate predictions or scores. Days 4, 6, and 7 selected hyperparameters on the same validation split used for Day 8, so the reported scores are susceptible to selection bias. Seed variability measures training subsampling on one fixed speaker split. The wider embedding branch can contribute more to RBF distances even after coordinate-wise standardization. A branch-weighting ablation and held-out test evaluation remain open.

Keep WAV files, feature archives, checkpoints, and locally fitted models out of Git. The repository tracks metadata and evaluation CSVs. See [.gitignore](.gitignore).

## References

- Livingstone & Russo (2018), [RAVDESS paper](https://doi.org/10.1371/journal.pone.0196391) and [dataset release](https://zenodo.org/records/1188976).
- Ma et al., [emotion2vec paper](https://arxiv.org/abs/2312.15185), [emotion2vec_base model card](https://huggingface.co/emotion2vec/emotion2vec_base), and [FunASR implementation](https://github.com/modelscope/FunASR/blob/main/funasr/models/emotion2vec/model.py).
