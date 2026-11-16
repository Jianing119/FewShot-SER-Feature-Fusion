from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "data" / "metadata.csv"

FEATURE_DIR = ROOT / "features"
FEATURE_PATH = FEATURE_DIR / "mfcc_features.npz"

SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 400
HOP_LENGTH = 160


def extract_mfcc(audio_path):
    """
    Extract 80-dimensional MFCC features
    from one audio file.
    """

    # Load and resample audio to 16 kHz
    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if sr != SAMPLE_RATE or len(y) == 0:
        raise ValueError(f"Invalid audio: {audio_path}")

    # Extract 40 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=80
    )

    # Statistics across time frames
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 40 means + 40 standard deviations
    features = np.concatenate([
        mfcc_mean,
        mfcc_std
    ]).astype(np.float32)

    if features.shape != (80,):
        raise ValueError(
            f"Unexpected feature shape: {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            f"Invalid feature values: {audio_path}"
        )

    return features


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    df = pd.read_csv(
        METADATA_PATH,
        dtype={"emotion_code": str}
    )

    assert len(df) == 768
    assert df["file_path"].is_unique

    print(f"Loaded metadata: {len(df)} files")
    print("Extracting MFCC features...")

    all_features = []

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df)
    ):
        audio_path = ROOT / row.file_path

        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        feature = extract_mfcc(audio_path)
        all_features.append(feature)

    X = np.stack(all_features).astype(np.float32)

    labels = df["emotion"].to_numpy(dtype=str)
    splits = df["split"].to_numpy(dtype=str)
    file_paths = df["file_path"].to_numpy(dtype=str)

    assert X.shape == (768, 80)
    assert np.isfinite(X).all()
    assert len(labels) == len(X)
    assert len(splits) == len(X)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        FEATURE_PATH,
        X=X,
        labels=labels,
        splits=splits,
        file_paths=file_paths
    )

    print("\n===== MFCC Extraction Completed =====")
    print("Feature matrix shape:", X.shape)
    print("Data type:", X.dtype)
    print("NaN count:", np.isnan(X).sum())
    print("Inf count:", np.isinf(X).sum())
    print("Labels shape:", labels.shape)
    print("Splits shape:", splits.shape)
    print("Saved to:", FEATURE_PATH)


if __name__ == "__main__":
    main()
