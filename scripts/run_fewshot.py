"""Reproduce Day 8 RAVDESS validation-only few-shot experiments."""

import argparse
import csv
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABELS = ("happy", "sad", "angry", "fearful")
SEEDS = (42, 123, 2026, 3407, 777)
SHOTS = (5, 10, 20, 50, 128)
PARAMETERS = {"mfcc": (100, 0.01), "emotion2vec": (10, 0.001), "fusion": (10, 0.001)}
METRICS = ("accuracy", "macro_f1", "uar")
SPEAKERS = {"train": range(1, 17), "val": range(17, 21), "test": range(21, 25)}
METADATA = ("labels", "splits", "file_paths")


def load_archive(path, width):
    with np.load(path, allow_pickle=False) as archive:
        features = archive["X"]
        metadata = tuple(archive[key] for key in METADATA)
    if (
        features.shape != (768, width)
        or features.dtype.kind not in "fiu"
        or not np.isfinite(features).all()
    ):
        raise ValueError(f"{path}: expected a finite real matrix of shape (768, {width})")
    if any(field.shape != (768,) or field.dtype.kind not in "US" for field in metadata):
        raise ValueError(f"{path}: expected 768 strings in each metadata field")
    return features, tuple(field.astype(str) for field in metadata)


def validate_metadata(metadata):
    labels, splits, paths = metadata
    if set(labels) != set(LABELS) or set(splits) != set(SPEAKERS):
        raise ValueError("Unexpected emotion labels or split names")
    # RAVDESS speech file names encode emotion and actor in fields 3 and 7.
    names = [path.replace(chr(92), "/").rsplit("/", 1)[-1].lower() for path in paths]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate RAVDESS file names")
    emotion_codes = dict(zip(("03", "04", "05", "06"), LABELS))
    for row, (name, label, split) in enumerate(zip(names, labels, splits)):
        match = re.fullmatch(r"(\d{2}-){6}\d{2}\.wav", name)
        if match is None:
            raise ValueError(f"Row {row}: invalid RAVDESS WAV name")
        fields = name[:-4].split("-")
        if int(fields[6]) not in SPEAKERS[split] or emotion_codes.get(fields[2]) != label:
            raise ValueError(f"Row {row}: WAV actor or emotion conflicts with metadata")
    for split, actors in SPEAKERS.items():
        for label in LABELS:
            count = np.count_nonzero((splits == split) & (labels == label))
            if count != 8 * len(actors):
                raise ValueError(f"{split}/{label}: unexpected count {count}")


def evaluate(method, train_x, train_y, val_x, val_y):
    c, gamma = PARAMETERS[method]
    scaler = StandardScaler()
    if method == "fusion":
        # Per-coordinate scaling preserves Day 7's weighting. The 768-D
        # branch can contribute more squared distance than the 80-D branch.
        scaler = ColumnTransformer(
            [
                ("mfcc", StandardScaler(), slice(0, 80)),
                ("emotion2vec", StandardScaler(), slice(80, 848)),
            ]
        )
    model = make_pipeline(scaler, SVC(kernel="rbf", C=c, gamma=gamma))
    model.fit(train_x, train_y)
    predicted = model.predict(val_x)
    return {
        "accuracy": accuracy_score(val_y, predicted),
        "macro_f1": f1_score(val_y, predicted, labels=LABELS, average="macro", zero_division=0),
        "uar": recall_score(val_y, predicted, labels=LABELS, average="macro", zero_division=0),
    }


def run(root):
    features_dir = root / "features"
    mfcc, metadata = load_archive(features_dir / "mfcc_features.npz", 80)
    emotion2vec, other_metadata = load_archive(features_dir / "emotion2vec_features.npz", 768)
    for key, left, right in zip(METADATA, metadata, other_metadata):
        if not np.array_equal(left, right):
            raise ValueError(f"Feature archives disagree on row order or {key}")
    validate_metadata(metadata)
    labels, splits, _ = metadata
    train_rows = np.flatnonzero(splits == "train")
    val_rows = np.flatnonzero(splits == "val")
    pools = [train_rows[labels[train_rows] == label] for label in LABELS]
    feature_sets = {
        "mfcc": (mfcc[train_rows], mfcc[val_rows]),
        "emotion2vec": (emotion2vec[train_rows], emotion2vec[val_rows]),
    }
    feature_sets["fusion"] = tuple(
        np.concatenate((left, right), axis=1)
        for left, right in zip(feature_sets["mfcc"], feature_sets["emotion2vec"])
    )
    train_y, val_y = labels[train_rows], labels[val_rows]
    pool_positions = [np.searchsorted(train_rows, pool) for pool in pools]
    records = []
    full_scores = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        permutations = [rng.permutation(pool) for pool in pool_positions]
        for shots in SHOTS:
            # Prefixes make budgets nested within a seed. Sort to restore the
            # original feature order, including the canonical full-train case.
            selected = np.sort(np.concatenate([order[:shots] for order in permutations]))
            for method, (train_x, val_x) in feature_sets.items():
                full_train = shots == SHOTS[-1]
                reused = full_train and method in full_scores
                if reused:
                    scores = full_scores[method]
                else:
                    scores = evaluate(method, train_x[selected], train_y[selected], val_x, val_y)
                    if full_train:
                        full_scores[method] = scores
                c, gamma = PARAMETERS[method]
                records.append({
                    "split": "val", "method": method, "shots_per_class": shots,
                    "seed": seed, "n_train": selected.size, "C": c, "gamma": gamma,
                    "reused_full_train": int(reused), **scores,
                })
    return records


def summarize(records):
    summary = []
    for method in PARAMETERS:
        for shots in SHOTS:
            group = [row for row in records if row["method"] == method and row["shots_per_class"] == shots]
            fits = [row for row in group if not row["reused_full_train"]]
            result = {
                "split": "val", "method": method, "shots_per_class": shots,
                "n_train": group[0]["n_train"], "n_records": len(group), "n_fits": len(fits),
            }
            for metric in METRICS:
                values = [row[metric] for row in fits]
                result[f"{metric}_mean"] = float(np.mean(values))
                result[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary.append(result)
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            writer = csv.DictWriter(temporary, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            temporary.close()
            temporary_path.replace(path)
        finally:
            temporary.close()
            if temporary_path.exists():
                temporary_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root
    records = run(root)
    summary = summarize(records)
    results = root / "results"
    write_csv(results / "fewshot_runs_val.csv", records)
    write_csv(results / "fewshot_summary_val.csv", summary)
    print(f"Wrote {len(records)} validation records and {len(summary)} summary rows to {results}")


if __name__ == "__main__":
    main()
