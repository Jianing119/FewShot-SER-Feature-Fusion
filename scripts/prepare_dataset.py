
from pathlib import Path
import pandas as pd
import soundfile as sf

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "RAVDESS"
OUTPUT = DATA_DIR / "metadata.csv"

EMOTIONS = {
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
}

TRAIN_ACTORS = set(range(1, 17))
VAL_ACTORS = set(range(17, 21))
TEST_ACTORS = set(range(21, 25))


def get_split(actor_id):
    if actor_id in TRAIN_ACTORS:
        return "train"
    if actor_id in VAL_ACTORS:
        return "val"
    if actor_id in TEST_ACTORS:
        return "test"
    raise ValueError(f"Unexpected actor ID: {actor_id}")


def parse_audio_file(path):
    parts = path.stem.split("-")

    if len(parts) != 7:
        raise ValueError(f"Invalid filename: {path.name}")

    modality, channel, emotion, intensity, statement, repetition, actor = parts

    # RAVDESS audio-only speech
    if modality != "03" or channel != "01":
        return None

    if emotion not in EMOTIONS:
        return None

    actor_id = int(actor)

    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Invalid audio: {path.name}")

    return {
        "file_name": path.name,
        "file_path": path.relative_to(ROOT).as_posix(),
        "actor_id": actor_id,
        "gender": "male" if actor_id % 2 else "female",
        "emotion": EMOTIONS[emotion],
        "emotion_code": emotion,
        "intensity": int(intensity),
        "statement": int(statement),
        "repetition": int(repetition),
        "sample_rate": info.samplerate,
        "duration_sec": round(
            info.frames / info.samplerate, 4
        ),
        "split": get_split(actor_id),
    }


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {RAW_DIR}"
        )

    wav_files = sorted(RAW_DIR.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")

    records = []

    for path in wav_files:
        record = parse_audio_file(path)
        if record is not None:
            records.append(record)

    if not records:
        raise RuntimeError("No matching speech audio found")

    df = pd.DataFrame(records)

    # Validate the expected complete dataset
    expected_emotions = set(EMOTIONS.values())
    assert set(df["emotion"]) == expected_emotions
    assert len(df) == 768, (
        f"Expected 768 files, found {len(df)}"
    )
    assert df["file_name"].is_unique
    assert set(df["actor_id"]) == set(range(1, 25))

    emotion_counts = df["emotion"].value_counts()
    assert (emotion_counts == 192).all()

    # Every actor should have 8 files per emotion
    actor_emotion_counts = (
        df.groupby(["actor_id", "emotion"])
        .size()
    )
    assert (actor_emotion_counts == 8).all()

    # Speaker-independent split validation
    actors = {
        name: set(group["actor_id"])
        for name, group in df.groupby("split")
    }

    assert actors["train"].isdisjoint(actors["val"])
    assert actors["train"].isdisjoint(actors["test"])
    assert actors["val"].isdisjoint(actors["test"])

    split_counts = df["split"].value_counts()
    assert split_counts["train"] == 512
    assert split_counts["val"] == 128
    assert split_counts["test"] == 128

    # Store relative paths for portability
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(
        ["actor_id", "emotion", "file_name"]
    ).reset_index(drop=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8")

    print("\nEmotion distribution:")
    print(df["emotion"].value_counts().sort_index())

    print("\nDataset split:")
    print(df["split"].value_counts())

    print("\nAudio information:")
    print(df[["sample_rate", "duration_sec"]].describe())

    print(f"\nMetadata saved to: {OUTPUT}")
    print("Dataset validation PASSED")


if __name__ == "__main__":
    main()
