"""
explore_data.py
===============
STEP 2: Explore and clean the scraped Korean movie review dataset.

This script:
  1. Loads data/naver_reviews.csv
  2. Reports basic statistics (shape, dtypes, nulls, duplicates)
  3. Cleans the data (removes nulls, duplicates, empty/too-short reviews)
  4. Shows score distribution (how many 1s, 2s, ... 10s)
  5. Saves a histogram plot → data/score_distribution.png
  6. Saves cleaned data → data/reviews_cleaned.csv

Run:
  python explore_data.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
RAW_CSV         = DATA_DIR / "naver_reviews.csv"
CLEANED_CSV     = DATA_DIR / "reviews_cleaned.csv"
PLOT_PATH       = DATA_DIR / "score_distribution.png"

# Minimum character length for a valid review
MIN_REVIEW_LEN  = 5


def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    return df


def report_quality(df: pd.DataFrame) -> None:
    """Print data quality statistics."""
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)

    # Missing values
    null_counts = df.isnull().sum()
    print("\n📌 Missing values per column:")
    for col, n in null_counts.items():
        pct = 100 * n / len(df)
        print(f"   {col}: {n:,}  ({pct:.1f}%)")

    # Duplicate rows
    n_dup = df.duplicated(subset=["review"]).sum()
    print(f"\n📌 Duplicate reviews: {n_dup:,}  ({100*n_dup/len(df):.1f}%)")

    # Score range
    if "score" in df.columns:
        print(f"\n📌 Score range: {df['score'].min()} – {df['score'].max()}")
        print(f"   Scores not in [1,10]: {((df['score'] < 1) | (df['score'] > 10)).sum():,}")

    # Review length
    if "review" in df.columns:
        df["_len"] = df["review"].dropna().apply(len)
        print(f"\n📌 Review text length (chars):")
        print(f"   Min    : {df['_len'].min()}")
        print(f"   Max    : {df['_len'].max()}")
        print(f"   Mean   : {df['_len'].mean():.1f}")
        print(f"   Median : {df['_len'].median():.0f}")
        df.drop(columns=["_len"], inplace=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove nulls, duplicates, and problematic reviews."""
    print("\n" + "="*60)
    print("CLEANING DATA")
    print("="*60)
    original_n = len(df)

    # Drop rows with null review or score
    df = df.dropna(subset=["review", "score"])
    print(f"  After removing nulls       : {len(df):,} rows")

    # Ensure score is integer in [1, 10]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    df["score"] = df["score"].astype(int)
    df = df[(df["score"] >= 1) & (df["score"] <= 10)]
    print(f"  After fixing score range   : {len(df):,} rows")

    # Clean review text
    df["review"] = df["review"].astype(str).str.strip()

    # Remove very short reviews (less than MIN_REVIEW_LEN chars)
    df = df[df["review"].str.len() >= MIN_REVIEW_LEN]
    print(f"  After removing short texts : {len(df):,} rows")

    # Remove duplicates based on review text only
    df = df.drop_duplicates(subset=["review"])
    print(f"  After removing duplicates  : {len(df):,} rows")

    # Reset index
    df = df.reset_index(drop=True)

    removed = original_n - len(df)
    print(f"\n✓ Removed {removed:,} problematic rows")
    print(f"✓ Final clean dataset: {len(df):,} rows")

    return df


def show_distribution(df: pd.DataFrame) -> None:
    """Print score distribution table."""
    print("\n" + "="*60)
    print("SCORE DISTRIBUTION")
    print("="*60)
    counts = df["score"].value_counts().sort_index()
    total  = len(df)

    print(f"\n{'Score':>6} | {'Count':>8} | {'%':>6} | Bar")
    print("-" * 50)
    for score, count in counts.items():
        pct = 100 * count / total
        bar = "█" * int(pct / 0.8)  # Each █ ~ 0.8%
        print(f"{score:>6} | {count:>8,} | {pct:>5.1f}% | {bar}")
    print("-" * 50)
    print(f"{'TOTAL':>6} | {total:>8,} | {100.0:>5.1f}%")


def plot_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Create and save a histogram of score distribution."""
    print(f"\nSaving histogram → {save_path}")

    counts = df["score"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Korean Movie Review Score Distribution", fontsize=14, fontweight="bold")

    # Bar chart — count per score
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, 10))
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Score (1-10)")
    axes[0].set_ylabel("Number of Reviews")
    axes[0].set_title("Count per Score")
    axes[0].set_xticks(range(1, 11))
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for i, (score, count) in enumerate(counts.items()):
        axes[0].text(score, count + 50, f"{count:,}", ha="center", va="bottom", fontsize=8)

    # Pie chart — proportion
    labels  = [f"Score {s}" for s in counts.index]
    axes[1].pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 8},
    )
    axes[1].set_title("Proportion per Score")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Histogram saved: {save_path}")


def main():
    print("=" * 60)
    print("Korean Movie Review Dataset — EDA & Cleaning")
    print("=" * 60)

    if not RAW_CSV.exists():
        print(f"\n❌ ERROR: {RAW_CSV} not found.")
        print("   Please run scrape_reviews.py first.")
        return

    # Load
    df = load_data(RAW_CSV)

    # Quality report
    report_quality(df)

    # Clean
    df = clean_data(df)

    # Score distribution
    show_distribution(df)

    # Plot
    plot_distribution(df, PLOT_PATH)

    # Save cleaned data
    df.to_csv(CLEANED_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✓ Cleaned dataset saved → {CLEANED_CSV}")

    print("\n" + "="*60)
    print("DONE — EDA complete!")
    print("="*60)
    print(f"  Raw rows    : loaded from {RAW_CSV}")
    print(f"  Clean rows  : {len(df):,}")
    print(f"  Plot saved  : {PLOT_PATH}")
    print(f"  Clean CSV   : {CLEANED_CSV}")
    print("\nNext step → run: python prepare_data.py")


if __name__ == "__main__":
    main()
