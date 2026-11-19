"""Train the Day 7 early-fusion RBF SVM and select on validation only."""

import argparse
import csv
from pathlib import Path, PureWindowsPath

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("happy", "sad", "angry", "fearful")
EMOTION_CODES = dict(zip(("03", "04", "05", "06"), LABELS))
C_VALUES = (0.1, 1.0, 10.0, 100.0)
GAMMA_VALUES = (0.001, 0.01, 0.1)
SPLIT_COUNTS = {"train": 512, "val": 128, "test": 128}
ACTORS = {"train": set(range(1, 17)), "val": set(range(17, 21)), "test": set(range(21, 25))}
BRANCHES = {"mfcc": 80, "emotion2vec": 768}
METADATA = ("labels", "splits", "file_paths")


def load_archive(path, width):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = set((*METADATA, "X")) - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        metadata = {key: archive[key] for key in METADATA}
        features = archive["X"]

    if any(values.shape != (768,) or values.dtype.kind != "U" for values in metadata.values()):
        raise ValueError(f"{path}: expected 768 strings in each metadata field")
    if features.shape != (768, width) or features.dtype.kind != "f":
        raise ValueError(f"{path}: expected a 768 × {width} floating-point matrix")
    return features, metadata


def load_train_val(features_dir):
    mfcc, metadata = load_archive(features_dir / "mfcc_features.npz", BRANCHES["mfcc"])
    emotion2vec, other = load_archive(
        features_dir / "emotion2vec_features.npz", BRANCHES["emotion2vec"]
    )
    for key in METADATA:
        if not np.array_equal(metadata[key], other[key]):
            index = np.flatnonzero(metadata[key] != other[key])[0]
            raise ValueError(f"Archive alignment mismatch in {key} at row index {index}")

    labels, splits, paths = (metadata[key] for key in METADATA)
    if dict(zip(*np.unique(splits, return_counts=True))) != SPLIT_COUNTS:
        raise ValueError(f"Expected split counts {SPLIT_COUNTS}")

    # Audit all split assignments from metadata alone.
    seen = set()
    actors = {split: set() for split in SPLIT_COUNTS}
    for path, label, split in zip(paths, labels, splits):
        stem = PureWindowsPath(path).stem
        parts = stem.split("-")
        if len(parts) != 7 or EMOTION_CODES.get(parts[2]) != label or not parts[-1].isdigit():
            raise ValueError(f"Invalid filename or label: {path}")
        actor = int(parts[-1])
        if stem in seen or actor not in ACTORS[split]:
            raise ValueError(f"Duplicate utterance or invalid speaker assignment: {path} ({split})")
        seen.add(stem)
        actors[split].add(actor)
    if actors != ACTORS:
        raise ValueError("Actual speaker sets differ from the expected train/val/test sets")

    selected = []
    for split in ("train", "val"):
        mask = splits == split
        branches = {"mfcc": mfcc[mask], "emotion2vec": emotion2vec[mask]}
        targets = labels[mask]
        if set(targets) != set(LABELS) or any(
            not np.isfinite(values).all() for values in branches.values()
        ):
            raise ValueError(f"Missing class or non-finite features in {split}")
        selected.append((branches, targets))
    return selected


def transform_branches(branches, scalers, branch_order):
    return np.concatenate([scalers[name].transform(branches[name]) for name in branch_order], axis=1)


def select_model(train, val):
    train_branches, y_train = train
    val_branches, y_val = val
    scalers = {name: StandardScaler().fit(values) for name, values in train_branches.items()}
    for name, scaler in scalers.items():
        if scaler.n_samples_seen_ != len(y_train):
            raise RuntimeError(f"{name} scaler training count mismatch")
        np.testing.assert_allclose(scaler.mean_, train_branches[name].mean(axis=0, dtype=np.float64))

    X_train = transform_branches(train_branches, scalers, BRANCHES)
    X_val = transform_branches(val_branches, scalers, BRANCHES)

    rows = []
    best_key = None
    for c in C_VALUES:
        for gamma in GAMMA_VALUES:
            svm = SVC(kernel="rbf", C=c, gamma=gamma).fit(X_train, y_train)
            pred = svm.predict(X_val)
            row = {
                "split": "val",
                "C": c,
                "gamma": gamma,
                "accuracy": accuracy_score(y_val, pred),
                "macro_f1": f1_score(y_val, pred, labels=LABELS, average="macro", zero_division=0),
                "uar": recall_score(y_val, pred, labels=LABELS, average="macro", zero_division=0),
                "selected": 0,
            }
            rows.append(row)
            key = (row["macro_f1"], row["uar"], row["accuracy"], -c, -gamma)
            if best_key is None or key > best_key:
                best_key, best_row, best_svm, best_pred = key, row, svm, pred

    best_row["selected"] = 1
    bundle = {
        "scalers": scalers,
        "svm": best_svm,
        "branch_order": tuple(BRANCHES),
        "branch_dimensions": dict(BRANCHES),
    }
    print(f"Train {X_train.shape}, validation {X_val.shape}, candidates {len(rows)}")
    return rows, best_row, bundle, best_pred


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison_row(model, result):
    return {
        "model": model,
        **{key: result[key] for key in ("split", "C", "gamma", "accuracy", "macro_f1", "uar")},
    }


def read_baseline(path):
    with path.open(encoding="utf-8", newline="") as handle:
        selected = [row for row in csv.DictReader(handle) if row["selected"] == "1"]
    if len(selected) != 1 or selected[0]["split"] != "val":
        raise ValueError(f"{path}: expected exactly one selected validation row")
    return selected[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=ROOT / "features")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()

    comparison = [
        comparison_row("MFCC + SVM", read_baseline(args.baseline_dir / "mfcc_svm_val.csv")),
        comparison_row(
            "emotion2vec + SVM", read_baseline(args.baseline_dir / "emotion2vec_svm_val.csv")
        ),
    ]
    train, val = load_train_val(args.features_dir)
    rows, best_row, bundle, best_pred = select_model(train, val)
    comparison.append(comparison_row("MFCC + emotion2vec + SVM", best_row))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "fusion_svm.joblib"
    joblib.dump(bundle, model_path)
    saved = joblib.load(model_path)
    val_branches, _ = val
    saved_pred = saved["svm"].predict(
        transform_branches(val_branches, saved["scalers"], saved["branch_order"])
    )
    np.testing.assert_array_equal(saved_pred, best_pred)
    write_csv(args.results_dir / "fusion_svm_val.csv", rows)
    write_csv(args.results_dir / "model_comparison_val.csv", comparison)
    print(
        f"Selected validation: C={best_row['C']:g}, gamma={best_row['gamma']:g}, "
        f"Accuracy={best_row['accuracy']:.6f}, Macro-F1={best_row['macro_f1']:.6f}, "
        f"UAR={best_row['uar']:.6f}"
    )
    print(f"Results: {args.results_dir / 'fusion_svm_val.csv'}")
    print(f"Comparison: {args.results_dir / 'model_comparison_val.csv'}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
