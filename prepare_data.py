"""
prepare_data.py
================
STEP 3: Tokenize reviews and prepare train/val/test splits for model training.

This script:
  1. Loads data/reviews_cleaned.csv
  2. Tokenizes all reviews using the KoELECTRA tokenizer
  3. Splits into TRAIN (80%) / VALIDATION (10%) / TEST (10%)
  4. Saves tokenized tensors as plain dicts to data/ for fast reloading
     (plain dicts avoid torch.load pickling issues across modules)

Pretrained model used:
  monologg/koelectra-base-v3-discriminator
  - Trained on 34GB of Korean text
  - Great for movie reviews (general domain)
  - More suitable than KR-FinBert (which is finance-specific)

Run:
  python prepare_data.py
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import ElectraTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.monitor_interval = 0

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
CLEANED_CSV     = DATA_DIR / "reviews_cleaned.csv"
MODEL_NAME      = "monologg/koelectra-base-v3-discriminator"
MAX_LENGTH      = 256    # Max tokens per review (most reviews are shorter)
BATCH_SIZE      = 16     # Used when creating DataLoaders
RANDOM_STATE    = 42




def tokenize_reviews(texts: list[str], tokenizer: ElectraTokenizer) -> dict:
    """
    Tokenize a list of review strings.
    Returns a dict of tensors: input_ids, attention_mask.

    We process in batches for memory efficiency.
    """
    print(f"  Tokenizing {len(texts):,} reviews (max_length={MAX_LENGTH})...")

    CHUNK = 1000  # Process 1000 reviews at a time to avoid OOM
    all_input_ids      = []
    all_attention_mask = []

    for start in tqdm(
        range(0, len(texts), CHUNK),
        desc  = "  🟡 Tokenizing",
        unit  = "chunk",
        colour= "yellow",
        dynamic_ncols=True,
    ):
        chunk = texts[start : start + CHUNK]
        enc   = tokenizer(
            chunk,
            max_length      = MAX_LENGTH,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )
        all_input_ids.append(enc["input_ids"])
        all_attention_mask.append(enc["attention_mask"])

    return {
        "input_ids"      : torch.cat(all_input_ids,      dim=0),
        "attention_mask" : torch.cat(all_attention_mask, dim=0),
    }


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into 80/10/10 train/val/test preserving score distribution (stratify)."""
    # First split: 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df,
        test_size    = 0.20,
        random_state = RANDOM_STATE,
        stratify     = df["score"],   # Keep score distribution balanced
    )
    # Second split: 50/50 of temp → 10% val, 10% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size    = 0.50,
        random_state = RANDOM_STATE,
        stratify     = temp_df["score"],
    )
    return train_df, val_df, test_df


def main():
    print("=" * 60)
    print("Step 3: Data Preparation & Tokenization")
    print("=" * 60)
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Max tokens: {MAX_LENGTH}")
    print(f"  Split    : 80% train / 10% val / 10% test")

    # ── Load cleaned data ──────────────────────────────────────────────────
    if not CLEANED_CSV.exists():
        print(f"\n❌ ERROR: {CLEANED_CSV} not found.")
        print("   Please run explore_data.py first.")
        return

    print(f"\nLoading {CLEANED_CSV} ...")
    df = pd.read_csv(CLEANED_CSV, encoding="utf-8-sig")
    df["score"]  = df["score"].astype(int)
    df["review"] = df["review"].astype(str)
    print(f"  Loaded {len(df):,} reviews")

    # ── Split ──────────────────────────────────────────────────────────────
    print("\nSplitting into train / val / test ...")
    train_df, val_df, test_df = split_data(df)
    print(f"  Train : {len(train_df):,} reviews")
    print(f"  Val   : {len(val_df):,} reviews")
    print(f"  Test  : {len(test_df):,} reviews")

    # Verify score distributions are similar
    print("\n  Score distribution check (train | val | test):")
    for s in range(1, 11):
        tr = train_df["score"].value_counts().get(s, 0)
        v  = val_df["score"].value_counts().get(s, 0)
        te = test_df["score"].value_counts().get(s, 0)
        print(f"    Score {s}: {tr:>6,} | {v:>5,} | {te:>5,}")

    # ── Load tokenizer ─────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    print("(This may download from HuggingFace on first run...)")
    tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")

    # ── Tokenize each split ────────────────────────────────────────────────
    print("\n--- Tokenizing TRAIN ---")
    train_enc = tokenize_reviews(train_df["review"].tolist(), tokenizer)

    print("\n--- Tokenizing VALIDATION ---")
    val_enc   = tokenize_reviews(val_df["review"].tolist(), tokenizer)

    print("\n--- Tokenizing TEST ---")
    test_enc  = tokenize_reviews(test_df["review"].tolist(), tokenizer)

    # ── Save as plain dicts (avoids pickling issues when loading in other modules) ─
    def make_dict(enc, labels):
        return {
            "input_ids"      : enc["input_ids"],
            "attention_mask" : enc["attention_mask"],
            "labels"         : torch.tensor(labels, dtype=torch.float),
        }

    train_data = make_dict(train_enc, train_df["score"].tolist())
    val_data   = make_dict(val_enc,   val_df["score"].tolist())
    test_data  = make_dict(test_enc,  test_df["score"].tolist())

    # ── Save to disk ───────────────────────────────────────────────────────
    print("\nSaving tokenized datasets to disk ...")
    train_path = DATA_DIR / "train.pt"
    val_path   = DATA_DIR / "val.pt"
    test_path  = DATA_DIR / "test.pt"

    torch.save(train_data, train_path)
    torch.save(val_data,   val_path)
    torch.save(test_data,  test_path)

    n_tr = train_data["labels"].shape[0]
    n_v  = val_data["labels"].shape[0]
    n_te = test_data["labels"].shape[0]
    print(f"  ✓ Saved train → {train_path}  ({n_tr:,} samples)")
    print(f"  ✓ Saved val   → {val_path}   ({n_v:,} samples)")
    print(f"  ✓ Saved test  → {test_path}  ({n_te:,} samples)")

    # ── Quick sanity check ────────────────────────────────────────────────
    print("\nSanity check — first sample from train set:")
    print(f"  input_ids shape     : {train_data['input_ids'][0].shape}")
    print(f"  attention_mask shape: {train_data['attention_mask'][0].shape}")
    print(f"  label (score)       : {train_data['labels'][0].item():.0f}")

    print("\n" + "="*60)
    print("DONE — Data preparation complete!")
    print("="*60)
    print("\nNext step → run: python train_model.py --mode quick")


if __name__ == "__main__":
    main()
