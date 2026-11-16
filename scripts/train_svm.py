"""Fit and select the Day 4 MFCC baseline without evaluating the test split."""

import argparse
import csv
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("happy", "sad", "angry", "fearful")
EMOTION_CODES = dict(zip(("03", "04", "05", "06"), LABELS))
C_VALUES = (0.1, 1.0, 10.0, 100.0)
GAMMA_VALUES = (0.001, 0.01, 0.1)


def load_splits(feature_path):
    with np.load(feature_path, allow_pickle=False) as archive:
        splits = archive["splits"]
        if splits.shape != (768,) or dict(zip(*np.unique(splits, return_counts=True))) != {
            "train": 512, "val": 128, "test": 128
        }:
            raise ValueError("Expected train/val/test counts of 512/128/128.")

        X = archive["X"]
        labels = archive["labels"]
        paths = archive["file_paths"]
        if X.shape != (768, 80) or labels.shape != splits.shape or paths.shape != splits.shape:
            raise ValueError("Unexpected MFCC or row metadata shape.")

        data = {}
        seen = set()
        for split, expected_actors in (("train", set(range(1, 17))),
                                       ("val", set(range(17, 21)))):
            mask = splits == split
            features, targets = X[mask], labels[mask]
            if not np.isfinite(features).all() or set(targets) != set(LABELS):
                raise ValueError(f"Invalid features or label set in {split}.")

            actors = set()
            for path, target in zip(paths[mask], targets):
                stem = Path(path).stem
                parts = stem.split("-")
                if len(parts) != 7 or EMOTION_CODES.get(parts[2]) != target:
                    raise ValueError(f"Filename and label disagree: {path}")
                if stem in seen:
                    raise ValueError(f"Duplicate utterance: {path}")
                seen.add(stem)
                actors.add(int(parts[-1]))
            if actors != expected_actors:
                raise ValueError(f"Unexpected speaker assignment in {split}: {sorted(actors)}")
            data[split] = features, targets

    return data["train"], data["val"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "features" / "mfcc_features.npz")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()

    (X_train, y_train), (X_val, y_val) = load_splits(args.features)
    rows = []
    best_key, best_model, best_row = None, None, None
    for c in C_VALUES:
        for gamma in GAMMA_VALUES:
            model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c, gamma=gamma))
            model.fit(X_train, y_train)
            if not np.all(model.named_steps["standardscaler"].n_samples_seen_ == len(X_train)):
                raise RuntimeError("Scaler was not fitted on exactly the training rows.")

            pred = model.predict(X_val)
            row = {
                "split": "val",
                "C": c,
                "gamma": gamma,
                "accuracy": accuracy_score(y_val, pred),
                "macro_f1": f1_score(y_val, pred, labels=LABELS, average="macro", zero_division=0),
                "uar": recall_score(y_val, pred, labels=LABELS, average="macro", zero_division=0),
                "selected": 0,
            }
            # Maximize macro-F1, then UAR and accuracy; prefer smaller C and gamma on ties.
            key = (row["macro_f1"], row["uar"], row["accuracy"], -c, -gamma)
            if best_key is None or key > best_key:
                best_key, best_model, best_row = key, model, row
            rows.append(row)

    best_row["selected"] = 1
    scaler = best_model.named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0, dtype=np.float64))
    csv_path = args.results_dir / "mfcc_svm_val.csv"
    model_path = args.model_dir / "mfcc_svm.joblib"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    # Reload our own artifact and check that the saved pipeline reproduces the score.
    saved_pred = joblib.load(model_path).predict(X_val)
    if not math.isclose(accuracy_score(y_val, saved_pred), best_row["accuracy"]):
        raise RuntimeError("Saved model did not reproduce validation accuracy.")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Train: {len(X_train)}; validation: {len(X_val)}; candidates: {len(rows)}")
    print("Selected validation result:", best_row)
    print(f"CSV: {csv_path}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
