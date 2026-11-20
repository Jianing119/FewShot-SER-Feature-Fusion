import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "final_evaluation_protocol.json"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

LABELS = ["happy", "sad", "angry", "fearful"]

FEATURE_FILES = {
    "mfcc": ROOT / "features" / "mfcc_features.npz",
    "emotion2vec": ROOT / "features" / "emotion2vec_features.npz",
}

MODEL_FILES = {
    "mfcc": ROOT / "models" / "mfcc_svm.joblib",
    "emotion2vec": ROOT / "models" / "emotion2vec_svm.joblib",
    "fusion": ROOT / "models" / "fusion_svm.joblib",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path, expected):
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != expected:
        raise RuntimeError(f"File changed since protocol freeze: {path}")


def load_verified_data(protocol):
    archives = {}

    for name, path in FEATURE_FILES.items():
        verify_hash(path, protocol["feature_sha256"][name])

        with np.load(path, allow_pickle=False) as f:
            archives[name] = {
                key: f[key]
                for key in ("X", "labels", "splits", "file_paths")
            }

    mfcc = archives["mfcc"]
    e2v = archives["emotion2vec"]

    if mfcc["X"].shape != (768, 80):
        raise ValueError("MFCC shape mismatch")

    if e2v["X"].shape != (768, 768):
        raise ValueError("emotion2vec shape mismatch")

    for name in ("mfcc", "emotion2vec"):
        if not np.isfinite(archives[name]["X"]).all():
            raise ValueError(f"{name} contains NaN or Inf")

    for field in ("labels", "splits", "file_paths"):
        if not np.array_equal(mfcc[field], e2v[field]):
            raise ValueError(f"Metadata mismatch: {field}")

    splits = mfcc["splits"]
    paths = mfcc["file_paths"]

    expected = {"train": 512, "val": 128, "test": 128}
    actual = dict(zip(*np.unique(splits, return_counts=True)))

    if actual != expected:
        raise ValueError(f"Unexpected split counts: {actual}")

    mask = splits == "test"
    actors = {
        int(Path(path).stem.split("-")[-1])
        for path in paths[mask]
    }

    if actors != set(protocol["test_actors"]):
        raise ValueError("Unexpected held-out test speakers")

    for name, path in MODEL_FILES.items():
        verify_hash(path, protocol["models"][name]["model_sha256"])

    return archives, mask


def predict_fusion(bundle, mfcc, e2v):
    if not isinstance(bundle, dict):
        raise TypeError("Unexpected fusion model format")

    branches = {
        "mfcc": mfcc,
        "emotion2vec": e2v,
    }

    order = tuple(bundle["branch_order"])

    if order != ("mfcc", "emotion2vec"):
        raise ValueError("Unexpected fusion branch order")

    transformed = [
        bundle["scalers"][name].transform(branches[name])
        for name in order
    ]

    X = np.concatenate(transformed, axis=1)

    if X.shape[1] != 848:
        raise ValueError("Fusion width mismatch")

    return bundle["svm"].predict(X)


def evaluate(archives, mask):
    mfcc = archives["mfcc"]
    e2v = archives["emotion2vec"]

    y_true = mfcc["labels"][mask]
    paths = mfcc["file_paths"][mask]

    counts = {label: int((y_true == label).sum())
              for label in LABELS}
    if counts != {label: 32 for label in LABELS}:
        raise ValueError(f"Unexpected test class distribution: {counts}")

    X_mfcc = mfcc["X"][mask]
    X_e2v = e2v["X"][mask]

    models = {
        name: joblib.load(path)
        for name, path in MODEL_FILES.items()
    }

    predictions = {
        "mfcc": models["mfcc"].predict(X_mfcc),
        "emotion2vec": models["emotion2vec"].predict(X_e2v),
        "fusion": predict_fusion(
            models["fusion"], X_mfcc, X_e2v
        ),
    }

    metric_rows = []
    class_rows = []

    prediction_table = pd.DataFrame({
        "file_path": paths,
        "true_emotion": y_true,
    })

    FIGURES.mkdir(parents=True, exist_ok=True)

    for name, y_pred in predictions.items():
        if not set(y_pred).issubset(set(LABELS)):
            raise ValueError(f"Unexpected predictions: {name}")

        prediction_table[f"{name}_pred"] = y_pred

        metrics = {
            "model": name,
            "split": "test",
            "n_samples": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true, y_pred, labels=LABELS,
                average="macro", zero_division=0
            ),
            "uar": recall_score(
                y_true, y_pred, labels=LABELS,
                average="macro", zero_division=0
            ),
        }

        metric_rows.append(metrics)

        report = classification_report(
            y_true, y_pred, labels=LABELS,
            output_dict=True, zero_division=0
        )

        for emotion in LABELS:
            class_rows.append({
                "model": name,
                "emotion": emotion,
                "precision": report[emotion]["precision"],
                "recall": report[emotion]["recall"],
                "f1_score": report[emotion]["f1-score"],
                "support": int(report[emotion]["support"]),
            })

        matrix = confusion_matrix(
            y_true, y_pred, labels=LABELS
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=LABELS
        ).plot(ax=ax, colorbar=False, values_format="d")

        ax.set_title(f"{name} - Held-out Test")
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"test_confusion_{name}.png",
            dpi=200
        )
        plt.close(fig)

    RESULTS.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metric_rows).to_csv(
        RESULTS / "heldout_metrics.csv", index=False
    )
    pd.DataFrame(class_rows).to_csv(
        RESULTS / "heldout_per_class.csv", index=False
    )
    prediction_table.to_csv(
        RESULTS / "heldout_predictions.csv", index=False
    )

    print("\n===== HELD-OUT EVALUATION COMPLETED =====")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nSaved 3 CSV files and 3 confusion matrices.")
    print("Frozen models used; no training or retuning.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="store_true",
        help="Perform the final held-out evaluation"
    )
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    if protocol["test_policy"] != "one final evaluation; no retuning":
        raise RuntimeError("Unexpected evaluation policy")

    archives, mask = load_verified_data(protocol)

    print("Protocol and file hashes: PASSED")
    print("Feature alignment: PASSED")
    print("Held-out speakers: PASSED")
    print("Test rows:", int(mask.sum()))

    if not args.run:
        print("Preflight PASSED. No predictions generated.")
        return

    output_paths = [
        RESULTS / "heldout_metrics.csv",
        RESULTS / "heldout_per_class.csv",
        RESULTS / "heldout_predictions.csv",
    ]
    output_paths += [
        FIGURES / f"test_confusion_{name}.png"
        for name in MODEL_FILES
    ]

    if any(path.exists() for path in output_paths):
        raise RuntimeError(
            "Existing held-out results found. "
            "Refusing to overwrite final evaluation."
        )

    evaluate(archives, mask)


if __name__ == "__main__":
    main()
