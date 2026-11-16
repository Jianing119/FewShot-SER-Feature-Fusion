from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

feature_path = ROOT / "features" / "mfcc_features.npz"
metadata_path = ROOT / "data" / "metadata.csv"

df = pd.read_csv(metadata_path)

with np.load(feature_path, allow_pickle=False) as data:
    X = data["X"]
    labels = data["labels"]
    splits = data["splits"]
    paths = data["file_paths"]

assert X.shape == (768, 80)
assert np.isfinite(X).all()

# Check row alignment with metadata
assert np.array_equal(
    labels,
    df["emotion"].to_numpy(dtype=str)
)

assert np.array_equal(
    splits,
    df["split"].to_numpy(dtype=str)
)

assert np.array_equal(
    paths,
    df["file_path"].to_numpy(dtype=str)
)

print("Feature shape:", X.shape)
print("Feature min:", X.min())
print("Feature max:", X.max())

print("\nSplit counts:")
for split in ["train", "val", "test"]:
    print(split, int(np.sum(splits == split)))

print("\nFeature and metadata alignment PASSED")
