import csv
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = ROOT / "features" / "emotion2vec_features.npz"
MFCC_PATH = ROOT / "features" / "mfcc_features.npz"

RESULT_PATH = ROOT / "results" / "emotion2vec_svm_val.csv"
MODEL_PATH = ROOT / "models" / "emotion2vec_svm.joblib"

LABELS = ("happy", "sad", "angry", "fearful")
C_VALUES = (0.1, 1.0, 10.0, 100.0)
GAMMA_VALUES = (0.001, 0.01, 0.1)


def load_splits():
    # Load pretrained speech features
    with np.load(FEATURE_PATH, allow_pickle=False) as data:
        X = data["X"]
        labels = data["labels"]
        splits = data["splits"]
        paths = data["file_paths"]

    if X.shape != (768, 768):
        raise ValueError(f"Unexpected feature shape: {X.shape}")

    if X.dtype != np.float32 or not np.isfinite(X).all():
        raise ValueError("Invalid feature dtype or values")

    # Verify alignment with the MFCC baseline
    with np.load(MFCC_PATH, allow_pickle=False) as data:
        for key, current in [
            ("labels", labels),
            ("splits", splits),
            ("file_paths", paths),
        ]:
            if not np.array_equal(current, data[key]):
                raise ValueError(f"Mismatch with MFCC: {key}")

    expected_counts = {
        "train": 512,
        "val": 128,
        "test": 128,
    }
    values, counts = np.unique(splits, return_counts=True)

    if dict(zip(values, counts)) != expected_counts:
        raise ValueError("Unexpected dataset split counts")

    expected_actors = {
        "train": set(range(1, 17)),
        "val": set(range(17, 21)),
    }

    for split in ("train", "val"):
        mask = splits == split
        actors = set()

        for path, label in zip(paths[mask], labels[mask]):
            parts = Path(path).stem.split("-")

            if len(parts) != 7:
                raise ValueError(f"Invalid filename: {path}")

            actor_id = int(parts[-1])
            actors.add(actor_id)

            emotion_codes = {
                "03": "happy",
                "04": "sad",
                "05": "angry",
                "06": "fearful",
            }

            if emotion_codes.get(parts[2]) != label:
                raise ValueError(f"Label mismatch: {path}")

        if actors != expected_actors[split]:
            raise ValueError(f"Speaker leakage or split error: {split}")

        if set(labels[mask]) != set(LABELS):
            raise ValueError(f"Missing emotion in {split}")

    train = splits == "train"
    val = splits == "val"

    return X[train], labels[train], X[val], labels[val]


def main():
    X_train, y_train, X_val, y_val = load_splits()

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test data: reserved, not evaluated")

    rows = []
    best_key = None
    best_model = None
    best_row = None

    for c in C_VALUES:
        for gamma in GAMMA_VALUES:
            model = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=c, gamma=gamma),
            )

            # Fit ONLY on training data
            model.fit(X_train, y_train)

            scaler = model.named_steps["standardscaler"]

            if scaler.n_samples_seen_ != len(X_train):
                raise RuntimeError("Scaler training count mismatch")

            # Predict ONLY on validation data
            pred = model.predict(X_val)

            row = {
                "split": "val",
                "C": c,
                "gamma": gamma,
                "accuracy": accuracy_score(y_val, pred),
                "macro_f1": f1_score(
                    y_val, pred,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                ),
                "uar": recall_score(
                    y_val, pred,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                ),
                "selected": 0,
            }

            rows.append(row)

            print(
                f"C={c:<6} gamma={gamma:<6} "
                f"Macro-F1={row['macro_f1']:.4f} "
                f"Accuracy={row['accuracy']:.4f}"
            )

            # Same selection rule as Day 4
            key = (
                row["macro_f1"],
                row["uar"],
                row["accuracy"],
                -c,
                -gamma,
            )

            if best_key is None or key > best_key:
                best_key = key
                best_model = model
                best_row = row

    best_row["selected"] = 1

    # Confirm scaler statistics use training data
    scaler = best_model.named_steps["standardscaler"]
    np.testing.assert_allclose(
        scaler.mean_,
        X_train.mean(axis=0, dtype=np.float64),
    )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    # Confirm the saved model reproduces predictions
    saved_model = joblib.load(MODEL_PATH)
    saved_pred = saved_model.predict(X_val)

    saved_f1 = f1_score(
        y_val, saved_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    if not math.isclose(saved_f1, best_row["macro_f1"]):
        raise RuntimeError("Saved model verification failed")

    with RESULT_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== Day 6 Completed =====")
    print("Candidates:", len(rows))
    print("Best C:", best_row["C"])
    print("Best gamma:", best_row["gamma"])
    print(f"Accuracy: {best_row['accuracy']:.4f}")
    print(f"Macro-F1: {best_row['macro_f1']:.4f}")
    print(f"UAR: {best_row['uar']:.4f}")
    print("Saved CSV:", RESULT_PATH)
    print("Saved model:", MODEL_PATH)


if __name__ == "__main__":
    main()
