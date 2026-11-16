"""Extract one emotion2vec embedding per RAVDESS metadata row."""

import argparse
import csv
from collections import Counter
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "metadata.csv"
OUTPUT = ROOT / "features" / "emotion2vec_features.npz"
CHECKPOINT = OUTPUT.with_name("emotion2vec_partial.npz")
MODEL = "emotion2vec/emotion2vec_base"
SHAPE = (768, 768)
SPLIT_COUNTS = {"train": 512, "val": 128, "test": 128}
SOURCE_RATE = 48000
TARGET_RATE = 16000


def load_metadata():
    with METADATA.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != SHAPE[0]:
        raise ValueError(f"{METADATA} must contain {SHAPE[0]} rows; got {len(rows)}")

    labels = []
    splits = []
    paths = []
    for line, row in enumerate(rows, start=2):
        label, split, path, rate = (
            row.get(key) for key in ("emotion", "split", "file_path", "sample_rate")
        )
        if label not in {"angry", "fearful", "happy", "sad"} or split not in SPLIT_COUNTS:
            raise ValueError(f"Invalid emotion or split at metadata line {line}")
        if not path or Path(path).suffix.lower() != ".wav" or rate != str(SOURCE_RATE):
            raise ValueError(f"Expected a WAV path and sample_rate={SOURCE_RATE} at metadata line {line}")
        labels.append(label)
        splits.append(split)
        paths.append(path)

    resolved_paths = [(ROOT / path).resolve() for path in paths]
    if len(set(resolved_paths)) != len(paths):
        raise ValueError("Duplicate audio paths in metadata")
    if Counter(splits) != SPLIT_COUNTS:
        raise ValueError(f"Unexpected split counts: {dict(Counter(splits))}")
    fields = {"labels": labels, "splits": splits, "file_paths": paths}
    return {key: np.asarray(values, dtype=str) for key, values in fields.items()}


def read_features(path, metadata, *, partial=False):
    with np.load(path, allow_pickle=False) as archive:
        matrix = archive["X"]
        if matrix.ndim != 2 or matrix.shape[1] != SHAPE[1] or matrix.dtype != np.float32:
            raise ValueError(f"{path.name}: invalid feature shape or dtype: {matrix.shape}, {matrix.dtype}")
        count = len(matrix)
        if not 0 < count <= SHAPE[0] or (not partial and count != SHAPE[0]):
            raise ValueError(f"{path.name}: invalid row count {count}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{path.name}: features contain NaN or Inf")
        for key, expected in metadata.items():
            if not np.array_equal(archive[key], expected[:count]):
                raise ValueError(f"{path.name}: {key} differs from metadata in row order or value")
    return matrix


def save_features(path, matrix, metadata):
    fields = {key: values[:len(matrix)] for key, values in metadata.items()}
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temp_path = Path(stream.name)
    try:
        np.savez_compressed(temp_path, X=matrix, **fields)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def extract_embedding(model, path):
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SOURCE_RATE:
        raise ValueError(f"WAV reports {rate} Hz, expected {SOURCE_RATE} Hz")
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError("Empty WAV or non-finite audio samples")
    mono = audio.mean(axis=1, dtype=np.float32)
    waveform = resample_poly(mono, up=1, down=SOURCE_RATE // TARGET_RATE)
    if not np.isfinite(waveform).all():
        raise ValueError("Resampled audio contains NaN or Inf")
    result = model.generate(
        input=waveform, fs=TARGET_RATE, granularity="utterance", extract_embedding=True
    )
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict) or "feats" not in result[0]:
        raise ValueError("Expected one FunASR result containing feats")
    embedding = np.asarray(result[0]["feats"], dtype=np.float32)
    if embedding.shape != (SHAPE[1],) or not np.isfinite(embedding).all():
        raise ValueError(f"Invalid embedding: shape {embedding.shape}, expected ({SHAPE[1]},) with no NaN or Inf")
    return embedding


def extract(metadata):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.empty(SHAPE, dtype=np.float32)
    completed = 0
    if CHECKPOINT.exists():
        prefix = read_features(CHECKPOINT, metadata, partial=True)
        completed = len(prefix)
        matrix[:completed] = prefix
        print(f"Resuming at {completed}/{SHAPE[0]} from {CHECKPOINT.name}", flush=True)

    if completed < SHAPE[0]:
        from tqdm import tqdm

        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            import torch
            from funasr import AutoModel

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in this Python environment")
            model = AutoModel(
                model=MODEL, hub="hf", device="cuda:0", disable_update=True, disable_pbar=True
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load {MODEL} on cuda:0. Use the Day 1 .venv and check CUDA and "
                f"HF_HUB_DISABLE_XET=1 / the Hugging Face model cache. {error}"
            ) from error

        saved = completed
        try:
            for index in tqdm(range(completed, SHAPE[0]), initial=completed, total=SHAPE[0],
                              desc="emotion2vec", unit="wav"):
                path = ROOT / metadata["file_paths"][index]
                try:
                    matrix[index] = extract_embedding(model, path)
                except Exception as error:
                    raise RuntimeError(f"Metadata line {index + 2}, WAV {path}: {error}") from error
                completed = index + 1
                if completed % 16 == 0:
                    save_features(CHECKPOINT, matrix[:completed], metadata)
                    saved = completed
        except (Exception, KeyboardInterrupt):
            if completed > saved:
                save_features(CHECKPOINT, matrix[:completed], metadata)
            raise

    save_features(OUTPUT, matrix, metadata)
    verify(metadata)
    CHECKPOINT.unlink(missing_ok=True)


def verify(metadata):
    matrix = read_features(OUTPUT, metadata)
    print(f"Verified {OUTPUT}")
    print(f"X shape: {matrix.shape}, dtype: {matrix.dtype}, NaN: 0, Inf: 0")
    print(f"Splits: {dict(Counter(metadata['splits'].tolist()))}")
    print("Row alignment PASSED (labels, splits, file_paths)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check the saved NPZ without running FunASR.")
    args = parser.parse_args()
    try:
        metadata = load_metadata()
        if args.verify_only or OUTPUT.exists():
            verify(metadata)
        else:
            extract(metadata)
    except KeyboardInterrupt:
        parser.exit(130, "\nInterrupted. Rerun the same command to continue from the last saved rows.\n")
    except Exception as error:
        print(f"Day 5 failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
