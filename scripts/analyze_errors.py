from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "features"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
FIGURE_DIR = RESULT_DIR / "figures"

LABELS = ["happy", "sad", "angry", "fearful"]

EXPECTED_F1 = {
    "mfcc": 0.621694,
    "emotion2vec": 0.676197,
    "fusion": 0.698886,
}


def load_features(filename, expected_width):
    with np.load(FEATURE_DIR / filename, allow_pickle=False) as f:
        X = f["X"]
        labels = f["labels"]
        splits = f["splits"]
        paths = f["file_paths"]

    if X.shape != (768, expected_width):
        raise ValueError(f"Wrong feature shape: {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError("NaN or Inf detected")

    return X, labels, splits, paths


def predict_fusion(bundle, X_mfcc, X_e2v):
    """
    Reconstruct the saved Day 7 fusion pipeline.
    """
    if not isinstance(bundle, dict):
        raise TypeError("Fusion model must be a saved bundle")

    order = bundle["branch_order"]
    scalers = bundle["scalers"]
    svm = bundle["svm"]

    if isinstance(scalers, dict):
        scaler_map = scalers
    else:
        scaler_map = dict(zip(order, scalers))

    branches = {
        "mfcc": X_mfcc,
        "emotion2vec": X_e2v,
    }

    transformed = []

    for name in order:
        if name not in branches or name not in scaler_map:
            raise ValueError(f"Unknown fusion branch: {name}")

        transformed.append(
            scaler_map[name].transform(branches[name])
        )

    X_fusion = np.concatenate(transformed, axis=1)

    if X_fusion.shape[1] != 848:
        raise ValueError("Fusion dimension must be 848")

    return svm.predict(X_fusion)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    X_mfcc, labels, splits, paths = load_features(
        "mfcc_features.npz", 80
    )

    X_e2v, e_labels, e_splits, e_paths = load_features(
        "emotion2vec_features.npz", 768
    )

    # Verify alignment
    for a, b in [
        (labels, e_labels),
        (splits, e_splits),
        (paths, e_paths),
    ]:
        if not np.array_equal(a, b):
            raise ValueError("Feature row alignment failed")

    # Validation only
    mask = splits == "val"

    if int(mask.sum()) != 128:
        raise ValueError("Expected 128 validation rows")

    y_true = labels[mask]
    val_paths = paths[mask]

    actors = {
        int(Path(path).stem.split("-")[-1])
        for path in val_paths
    }

    if actors != {17, 18, 19, 20}:
        raise ValueError("Unexpected validation speakers")

    mfcc_val = X_mfcc[mask]
    e2v_val = X_e2v[mask]

    # Load trusted local model artifacts only
    mfcc_model = joblib.load(
        MODEL_DIR / "mfcc_svm.joblib"
    )
    e2v_model = joblib.load(
        MODEL_DIR / "emotion2vec_svm.joblib"
    )
    fusion_bundle = joblib.load(
        MODEL_DIR / "fusion_svm.joblib"
    )

    predictions = {
        "mfcc": mfcc_model.predict(mfcc_val),
        "emotion2vec": e2v_model.predict(e2v_val),
        "fusion": predict_fusion(
            fusion_bundle, mfcc_val, e2v_val
        ),
    }

    metric_rows = []
    class_rows = []

    for name, pred in predictions.items():
        acc = accuracy_score(y_true, pred)
        macro_f1 = f1_score(
            y_true, pred, labels=LABELS,
            average="macro", zero_division=0
        )
        uar = recall_score(
            y_true, pred, labels=LABELS,
            average="macro", zero_division=0
        )

        if not np.isclose(
            macro_f1, EXPECTED_F1[name], atol=1e-4
        ):
            raise ValueError(
                f"{name} did not reproduce Day 4/6/7 "
                f"validation Macro-F1: {macro_f1:.6f}"
            )

        metric_rows.append({
            "model": name,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "uar": uar,
        })

        report = classification_report(
            y_true, pred, labels=LABELS,
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

        cm = confusion_matrix(
            y_true, pred, labels=LABELS
        )

        fig, ax = plt.subplots(figsize=(6.5, 5.5))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=LABELS
        )
        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format="d"
        )

        ax.set_title(f"{name} - Validation Confusion Matrix")
        fig.tight_layout()

        fig.savefig(
            FIGURE_DIR / f"confusion_{name}_val.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(fig)

    # Sample-level fusion comparison
    e2v_ok = predictions["emotion2vec"] == y_true
    fusion_ok = predictions["fusion"] == y_true

    status = np.select(
        [
            ~e2v_ok & fusion_ok,
            e2v_ok & ~fusion_ok,
            e2v_ok & fusion_ok,
        ],
        [
            "corrected_by_fusion",
            "regressed_by_fusion",
            "both_correct",
        ],
        default="both_wrong"
    )

    comparison = pd.DataFrame({
        "file_path": val_paths,
        "actor_id": [
            int(Path(p).stem.split("-")[-1])
            for p in val_paths
        ],
        "true_emotion": y_true,
        "mfcc_pred": predictions["mfcc"],
        "emotion2vec_pred": predictions["emotion2vec"],
        "fusion_pred": predictions["fusion"],
        "fusion_vs_e2v": status,
    })

    pd.DataFrame(metric_rows).to_csv(
        RESULT_DIR / "day10_model_metrics_val.csv",
        index=False
    )

    pd.DataFrame(class_rows).to_csv(
        RESULT_DIR / "day10_per_class_val.csv",
        index=False
    )

    comparison.to_csv(
        RESULT_DIR / "day10_error_samples_val.csv",
        index=False
    )

    print("\n===== Day 10 Completed =====")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nFusion vs emotion2vec:")
    print(comparison["fusion_vs_e2v"].value_counts())
    print("\nSaved 3 CSV tables and 3 confusion matrices")
    print("Test split was not evaluated")


if __name__ == "__main__":
    main()
