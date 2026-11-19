"""Day 9: validate and analyze RAVDESS speaker-independent few-shot validation runs.

Run from project root: python scripts/analyze_fewshot.py
Requires: pandas, numpy, matplotlib
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_PATH = ROOT / "results" / "fewshot_runs_val.csv"
EXISTING_SUMMARY_PATH = ROOT / "results" / "fewshot_summary_val.csv"
FIG_DIR = ROOT / "results" / "figures"
METHODS = ("mfcc", "emotion2vec", "fusion")
SHOTS = (5, 10, 20, 50, 128)
SEEDS = (42, 123, 2026, 3407, 777)
METRICS = ("accuracy", "macro_f1", "uar")
DISPLAY = {"mfcc": "MFCC", "emotion2vec": "emotion2vec", "fusion": "Fusion"}


def validate_runs(df):
    required = {"split", "method", "shots_per_class", "seed", "n_train", "reused_full_train", *METRICS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) != 75 or df[list(required)].isna().any().any():
        raise ValueError("Expected 75 complete validation records")
    expected = {(m, s, seed) for m in METHODS for s in SHOTS for seed in SEEDS}
    found = list(zip(df.method, df.shots_per_class, df.seed))
    if len(set(found)) != 75 or set(found) != expected:
        raise ValueError("Unexpected or duplicated method/shot/seed records")
    if set(df.split) != {"val"}:
        raise ValueError("Only validation results are allowed in Day 9")
    if not (df.n_train == df.shots_per_class * 4).all():
        raise ValueError("n_train must be 4 times shots_per_class")
    for metric in METRICS:
        if not df[metric].between(0, 1).all():
            raise ValueError(f"Metric out of range: {metric}")
    for method in METHODS:
        full = df[(df.method == method) & (df.shots_per_class == 128)]
        if full[list(METRICS)].drop_duplicates().shape[0] != 1:
            raise ValueError(f"Full-shot runs are not identical: {method}")
        if sorted(full.reused_full_train.tolist()) != [0, 1, 1, 1, 1]:
            raise ValueError(f"Expected 1 independent full-shot fit: {method}")
    if not (df[df.shots_per_class < 128].reused_full_train == 0).all():
        raise ValueError("Partial-shot runs must all be independent fits")


def analyze(df):
    summary_rows = []
    for method in METHODS:
        for shot in SHOTS:
            subset = df[(df.method == method) & (df.shots_per_class == shot)]
            unique_fits = subset[subset.reused_full_train == 0]
            row = {"method": method, "shots_per_class": shot,
                   "n_records": len(subset), "n_fits": len(unique_fits)}
            for metric in METRICS:
                values = unique_fits[metric].to_numpy(float)
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    pivot = df.pivot(index=["shots_per_class", "seed"], columns="method", values="macro_f1")
    if pivot.isna().any().any():
        raise ValueError("Cannot compare unpaired seeds")
    differences = (pivot["fusion"] - pivot["emotion2vec"]).rename("fusion_minus_e2v")
    paired_rows = []
    for shot in SHOTS:
        values = differences.loc[shot].to_numpy(float)
        if shot == 128:
            values = values[:1]  # only one genuinely independent full-shot fit
        paired_rows.append({
            "shots_per_class": shot,
            "n_independent_fits": len(values),
            "mean_diff_pp": 100 * values.mean(),
            "std_diff_pp": 100 * values.std(ddof=1) if len(values) > 1 else np.nan,
            "fusion_wins": int((values > 0).sum()),
            "emotion2vec_wins": int((values < 0).sum()),
            "ties": int((values == 0).sum()),
        })
    return summary, pd.DataFrame(paired_rows)


def check_existing_summary(summary):
    if not EXISTING_SUMMARY_PATH.exists():
        print("Previous Day 8 summary not found; continuing with raw runs")
        return
    old = pd.read_csv(EXISTING_SUMMARY_PATH)
    old = old.set_index(["method", "shots_per_class"])
    new = summary.set_index(["method", "shots_per_class"])
    for metric in METRICS:
        for suffix in ("mean", "std"):
            column = f"{metric}_{suffix}"
            if not np.allclose(old.loc[new.index, column], new[column], rtol=1e-7, atol=1e-10, equal_nan=True):
                raise ValueError(f"Day 8 summary disagrees with raw records: {column}")
    print("Day 8 summary consistency PASSED")


def create_figures(summary, paired):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    for method in METHODS:
        x = summary[summary.method == method].set_index("shots_per_class").loc[list(SHOTS)]
        xs = np.array(SHOTS)
        ys = x.macro_f1_mean.to_numpy() * 100
        std = x.macro_f1_std.to_numpy() * 100
        ax.plot(xs, ys, marker="o", label=DISPLAY[method])
        ax.errorbar(xs[:-1], ys[:-1], yerr=std[:-1], fmt="none", capsize=4)
    ax.set(xlabel="Training examples per emotion class", ylabel="Validation Macro-F1 (%)",
           title="Few-shot speech emotion recognition: mean and sample SD")
    ax.set_xticks(SHOTS)
    ax.set_ylim(0, 85)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fewshot_macro_f1.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xs = paired.shots_per_class.to_numpy()
    ys = paired.mean_diff_pp.to_numpy()
    errs = paired.std_diff_pp.to_numpy()
    ax.plot(xs, ys, marker="o")
    ax.errorbar(xs[:-1], ys[:-1], yerr=errs[:-1], fmt="none", capsize=4)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set(xlabel="Training examples per emotion class", ylabel="Fusion - emotion2vec Macro-F1 (pp)",
           title="Paired-seed difference on a fixed validation set")
    ax.set_xticks(SHOTS)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fusion_minus_emotion2vec.png", dpi=220)
    plt.close(fig)


def main():
    if not RUNS_PATH.exists():
        raise FileNotFoundError(f"Missing Day 8 runs: {RUNS_PATH}")
    df = pd.read_csv(RUNS_PATH)
    validate_runs(df)
    summary, paired = analyze(df)
    check_existing_summary(summary)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ROOT / "results" / "day9_summary_val.csv", index=False, float_format="%.10f")
    paired.to_csv(ROOT / "results" / "day9_paired_fusion_vs_e2v_val.csv", index=False, float_format="%.10f")
    create_figures(summary, paired)
    print("Day 9 validation checks PASSED")
    print("\nPaired Fusion - emotion2vec difference (percentage points):")
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nWrote 2 CSV tables and 2 PNG figures under results/")
    print("Note: validation only; no held-out test results were used.")


if __name__ == "__main__":
    main()
