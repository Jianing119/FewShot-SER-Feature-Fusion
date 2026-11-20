from pathlib import Path
import csv
import hashlib
import json

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "final_evaluation_protocol.json"

SETTINGS = {
    "mfcc": {
        "csv": "results/mfcc_svm_val.csv",
        "model": "models/mfcc_svm.joblib",
        "C": 100.0,
        "gamma": 0.01,
    },
    "emotion2vec": {
        "csv": "results/emotion2vec_svm_val.csv",
        "model": "models/emotion2vec_svm.joblib",
        "C": 10.0,
        "gamma": 0.001,
    },
    "fusion": {
        "csv": "results/fusion_svm_val.csv",
        "model": "models/fusion_svm.joblib",
        "C": 10.0,
        "gamma": 0.001,
    },
}

FEATURES = {
    "mfcc": "features/mfcc_features.npz",
    "emotion2vec": "features/emotion2vec_features.npz",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_scaler(scaler):
    counts = np.asarray(scaler.n_samples_seen_)
    if not np.all(counts == 512):
        raise ValueError("Scaler was not fitted on 512 rows")


def main():
    metadata = None

    for name, relative in FEATURES.items():
        with np.load(ROOT / relative, allow_pickle=False) as archive:
            X = archive["X"]
            width = 80 if name == "mfcc" else 768

            assert X.shape == (768, width)
            assert np.isfinite(X).all()

            current = [
                archive[k]
                for k in ("labels", "splits", "file_paths")
            ]

            if metadata is None:
                metadata = current
            else:
                for left, right in zip(metadata, current):
                    assert np.array_equal(left, right)

    labels, splits, paths = metadata

    assert dict(zip(*np.unique(splits, return_counts=True))) == {
        "train": 512,
        "val": 128,
        "test": 128,
    }

    for split, expected in (
        ("train", set(range(1, 17))),
        ("val", set(range(17, 21))),
        ("test", set(range(21, 25))),
    ):
        actors = {
            int(Path(p).stem.split("-")[-1])
            for p in paths[splits == split]
        }
        assert actors == expected

    locked = {}

    for name, setting in SETTINGS.items():
        with (ROOT / setting["csv"]).open(
            encoding="utf-8", newline=""
        ) as file:
            rows = list(csv.DictReader(file))

        selected = [r for r in rows if r["selected"] == "1"]
        assert len(rows) == 12
        assert len(selected) == 1

        row = selected[0]
        assert row["split"] == "val"
        assert np.isclose(float(row["C"]), setting["C"])
        assert np.isclose(float(row["gamma"]), setting["gamma"])

        model_path = ROOT / setting["model"]
        model = joblib.load(model_path)

        if name == "fusion":
            svm = model["svm"]
            assert tuple(model["branch_order"]) == (
                "mfcc", "emotion2vec"
            )
            for scaler in model["scalers"].values():
                check_scaler(scaler)
        else:
            svm = model.named_steps["svc"]
            check_scaler(model.named_steps["standardscaler"])

        assert np.isclose(svm.C, setting["C"])
        assert np.isclose(float(svm.gamma), setting["gamma"])

        locked[name] = {
            "C": setting["C"],
            "gamma": setting["gamma"],
            "validation_macro_f1": float(row["macro_f1"]),
            "model_sha256": sha256(model_path),
        }

        print(f"{name}: audit PASSED")

    protocol = {
        "version": 1,
        "dataset": "RAVDESS four emotions",
        "training_actors": list(range(1, 17)),
        "validation_actors": list(range(17, 21)),
        "test_actors": list(range(21, 25)),
        "primary_metric": "macro_f1",
        "secondary_metrics": ["accuracy", "uar"],
        "training_policy": "train-only 512; no refit",
        "test_policy": "one final evaluation; no retuning",
        "models": locked,
        "feature_sha256": {
            name: sha256(ROOT / path)
            for name, path in FEATURES.items()
        },
    }

    CONFIG.parent.mkdir(parents=True, exist_ok=True)

    if CONFIG.exists():
        previous = json.loads(CONFIG.read_text(encoding="utf-8"))
        if previous != protocol:
            raise RuntimeError(
                "Existing frozen protocol differs. "
                "Do not silently overwrite it."
            )
        print("Existing protocol verified")
    else:
        CONFIG.write_text(
            json.dumps(protocol, indent=2) + "\n",
            encoding="utf-8"
        )
        print("Final evaluation protocol created")

    print("Day 11 pre-test audit PASSED")
    print("No test predictions were generated")


if __name__ == "__main__":
    main()
