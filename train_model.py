"""
train_model.py
==============
STEP 4: Fine-tune KoELECTRA for Korean movie review score prediction.

Two training modes:
  1. REGRESSION    — predict a single continuous score (1-10) using MSE loss
  2. ORDINAL       — predict one of 10 integer score classes using CrossEntropy

Usage:
  python train_model.py --mode quick        # 5,000 samples, verify pipeline
  python train_model.py --mode full         # Full dataset, full training
  python train_model.py --approach regression
  python train_model.py --approach ordinal
  (Default: runs both approaches back-to-back)

Output:
  models/regression/  → best regression model checkpoint
  models/ordinal/     → best ordinal classification model checkpoint
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
from transformers import (
    ElectraModel,
    ElectraTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm
tqdm.monitor_interval = 0  # suppress monitor thread warnings on Windows
import json

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
MODEL_NAME  = "monologg/koelectra-base-v3-discriminator"

# Hyperparameters
LEARNING_RATE  = 2e-5     # Small LR needed for fine-tuning transformers
WEIGHT_DECAY   = 0.01     # Regularization to prevent overfitting
BATCH_SIZE     = 16       # Adjust down to 8 if you get out-of-memory errors
MAX_EPOCHS     = 5        # Number of passes over the training data
WARMUP_RATIO   = 0.1      # 10% of steps used for LR warm-up
QUICK_N        = 1_000    # Number of samples for the quick test run
QUICK_EPOCHS   = 1        # Just 1 epoch in quick mode to verify pipeline

NUM_CLASSES    = 10       # Scores 1-10
MAX_LENGTH     = 256      # Max tokens per review (must match prepare_data.py)


class ReviewDataset(Dataset):
    """
    PyTorch Dataset wrapping the tokenized review dict.
    Accepts a plain dict with keys: input_ids, attention_mask, labels.
    """

    def __init__(self, data: dict):
        self.input_ids      = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels         = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids"      : self.input_ids[idx],
            "attention_mask" : self.attention_mask[idx],
            "labels"         : self.labels[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────────────────────────────────────

class KoELECTRARegressor(nn.Module):
    """
    REGRESSION MODEL: KoELECTRA backbone + a single output neuron.

    Architecture:
      KoELECTRA encoder → [CLS] token embedding (768-dim)
      → Dropout (prevent overfitting)
      → Linear(768, 256) → GELU activation
      → Dropout
      → Linear(256, 1)   ← predicted score
      → Clamped to [1.0, 10.0]

    Loss: MSELoss (mean squared error between predicted and true score)
    """

    def __init__(self, model_name: str, dropout_rate: float = 0.3):
        super().__init__()
        # Load pretrained KoELECTRA encoder
        self.electra  = ElectraModel.from_pretrained(model_name)
        hidden_size   = self.electra.config.hidden_size  # 768 for base models

        # Regression head: 768 → 256 → 1
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        # Run through ELECTRA; output shape: (batch, seq_len, 768)
        outputs     = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token (index 0) — it represents the entire sentence
        cls_output  = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        # Predict score
        logits      = self.regressor(cls_output).squeeze(-1)  # (batch,)
        # Clamp to valid score range
        predictions = torch.clamp(logits, min=1.0, max=10.0)
        return predictions


class KoELECTRAOrdinalClassifier(nn.Module):
    """
    ORDINAL CLASSIFICATION MODEL: KoELECTRA backbone + 10-class head.

    Architecture:
      KoELECTRA encoder → [CLS] token embedding (768-dim)
      → Dropout
      → Linear(768, 256) → GELU
      → Dropout
      → Linear(256, 10)  ← 10 logits, one per score class (1-10)

    Loss: CrossEntropyLoss (treats each score as a distinct class)
    
    Note: We store scores as 0-indexed internally (score 1 = class 0, etc.)
    """

    def __init__(self, model_name: str, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        self.electra    = ElectraModel.from_pretrained(model_name)
        hidden_size     = self.electra.config.hidden_size

        # Classification head: 768 → 256 → 10
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs     = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        cls_output  = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        logits      = self.classifier(cls_output)          # (batch, 10)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Training Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    """Use CUDA (GPU) if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  CPU training — this will be slow for large datasets.")
        print("  Consider using Google Colab (free GPU) for full training.")
    return device


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, approach):
    """Run one full pass through the training data and return average loss."""
    model.train()       # Enable dropout (training mode)
    total_loss  = 0.0
    total_items = 0

    bar = tqdm(
        loader,
        desc  = f"  🔵 Training",
        unit  = "batch",
        colour= "cyan",
        dynamic_ncols=True,
        leave = False,
    )
    for batch in bar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()  # Clear gradients from previous step

        if approach == "regression":
            # Regression: predict continuous score
            predictions = model(input_ids, attention_mask)
            loss        = criterion(predictions, labels)

        else:  # ordinal classification
            # Classification: convert score (1-10) → class index (0-9)
            class_labels = (labels - 1).long()   # score 1 → index 0
            logits       = model(input_ids, attention_mask)
            loss         = criterion(logits, class_labels)

        # Backward pass: compute gradients
        loss.backward()

        # Gradient clipping: prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()   # Update model weights
        scheduler.step()   # Update learning rate

        total_loss  += loss.item() * len(labels)
        total_items += len(labels)
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_items


def evaluate_epoch(model, loader, criterion, device, approach):
    """Evaluate model on a dataset, return average loss and MAE."""
    model.eval()  # Disable dropout (evaluation mode)
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():  # No gradient computation needed during eval
        for batch in tqdm(
            loader,
            desc  = "  🟢 Evaluating",
            unit  = "batch",
            colour= "green",
            dynamic_ncols=True,
            leave = False,
        ):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            if approach == "regression":
                predictions = model(input_ids, attention_mask)
                loss        = criterion(predictions, labels)
                pred_scores = predictions.cpu().numpy()

            else:  # ordinal
                class_labels = (labels - 1).long()
                logits       = model(input_ids, attention_mask)
                loss         = criterion(logits, class_labels)
                # Convert class probabilities → score (argmax + 1 to get 1-10)
                pred_classes = torch.argmax(logits, dim=1)
                pred_scores  = (pred_classes + 1).float().cpu().numpy()

            total_loss  += loss.item() * len(labels)
            all_preds.extend(pred_scores.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    n    = len(all_labels)
    loss = total_loss / n
    mae  = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    return loss, mae


def save_step_checkpoint(save_dir, model, optimizer, scheduler,
                         epoch, global_step, best_val_mae, history):
    """Save a full resumable checkpoint mid-epoch."""
    ckpt = {
        "epoch"        : epoch,
        "global_step"  : global_step,
        "best_val_mae" : best_val_mae,
        "history"      : history,
        "model"        : model.state_dict(),
        "optimizer"    : optimizer.state_dict(),
        "scheduler"    : scheduler.state_dict(),
    }
    path = save_dir / "checkpoint_latest.pt"
    torch.save(ckpt, path)
    print(f"  💾 Checkpoint saved at step {global_step} → {path}")


def load_step_checkpoint(save_dir, model, optimizer, scheduler):
    """Load the latest step checkpoint if it exists. Returns (epoch, global_step, best_val_mae, history)."""
    path = save_dir / "checkpoint_latest.pt"
    if not path.exists():
        return 1, 0, float("inf"), []
    print(f"  Resuming from checkpoint: {path}")
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"    -> Epoch {ckpt['epoch']}, step {ckpt['global_step']}, best MAE {ckpt['best_val_mae']:.4f}")
    return ckpt["epoch"], ckpt["global_step"], ckpt["best_val_mae"], ckpt["history"]


def train_model(approach: str, quick_mode: bool, save_steps: int = 500):
    """Full training loop for regression or ordinal classification."""
    print("\n" + "="*60)
    print(f"TRAINING: {approach.upper()}")
    if quick_mode:
        print(f"  Mode: QUICK ({QUICK_N:,} samples -- for pipeline verification)")
    else:
        print(f"  Mode: FULL (all data)")
    print("="*60)

    save_dir = MODELS_DIR / approach
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Check if already fully trained ───────────────────────────────────
    complete_marker = save_dir / "training_complete.json"
    if complete_marker.exists() and not quick_mode:
        with open(complete_marker, encoding="utf-8") as f:
            info = json.load(f)
        print(f"  Training already completed for {approach}.")
        print(f"  Best Val MAE: {info['best_val_mae']:.4f} (from {info['epochs_run']} epochs)")
        print(f"  Delete '{complete_marker}' to re-train from scratch.")
        return info["best_val_mae"]

    device = get_device()

    # ── Load datasets ───────────────────────────────────
    print("\nLoading datasets ...")
    train_data = torch.load(DATA_DIR / "train.pt", weights_only=True)
    val_data   = torch.load(DATA_DIR / "val.pt",   weights_only=True)
    train_ds   = ReviewDataset(train_data)
    val_ds     = ReviewDataset(val_data)

    if quick_mode:
        n     = min(QUICK_N, len(train_ds))
        idxs  = torch.randperm(len(train_ds))[:n].tolist()
        train_ds = Subset(train_ds, idxs)
        n_val = min(1000, len(val_ds))
        idxs  = torch.randperm(len(val_ds))[:n_val].tolist()
        val_ds = Subset(val_ds, idxs)

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Build model ────────────────────────────────────────────────────────
    print(f"\nBuilding {approach} model from: {MODEL_NAME}")
    if approach == "regression":
        model     = KoELECTRARegressor(MODEL_NAME).to(device)
        criterion = nn.MSELoss()
    else:
        model     = KoELECTRAOrdinalClassifier(MODEL_NAME, NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()

    # ── Optimizer & Scheduler ──────────────────────────────────────────────
    no_decay    = ["bias", "LayerNorm.weight"]
    params      = [
        {
            "params"      : [p for n, p in model.named_parameters()
                             if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params"      : [p for n, p in model.named_parameters()
                             if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer    = AdamW(params, lr=LEARNING_RATE)
    n_epochs     = QUICK_EPOCHS if quick_mode else MAX_EPOCHS
    total_steps  = len(train_loader) * n_epochs
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Auto-detect checkpoint and resume ─────────────────────────────
    if (save_dir / "checkpoint_latest.pt").exists():
        start_epoch, global_step, best_val_mae, history = \
            load_step_checkpoint(save_dir, model, optimizer, scheduler)
    else:
        start_epoch, global_step, best_val_mae, history = 1, 0, float("inf"), []

    steps_per_epoch = len(train_loader)
    # How many batches to skip in the start epoch (already trained)
    start_batch = global_step % steps_per_epoch

    print(f"\nStarting training for {n_epochs} epoch(s) ...")
    print(f"  Optimizer  : AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"  Warmup     : {warmup_steps} steps")
    print(f"  Save dir   : {save_dir}")
    print(f"  Save every : {save_steps} steps")
    if start_batch > 0:
        print(f"  Resuming   : epoch {start_epoch}, skipping first {start_batch} batches")
    print()

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}")
        model.train()
        total_loss  = 0.0
        total_items = 0

        bar = tqdm(
            enumerate(train_loader),
            total = steps_per_epoch,
            desc  = f"  🔵 Training",
            unit  = "batch",
            colour= "cyan",
            dynamic_ncols=True,
            leave = False,
            initial = start_batch if epoch == start_epoch else 0,
        )

        for batch_idx, batch in bar:
            # Skip batches already trained in this epoch on resume
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            if approach == "regression":
                predictions = model(input_ids, attention_mask)
                loss        = criterion(predictions, labels)
            else:
                class_labels = (labels - 1).long()
                logits       = model(input_ids, attention_mask)
                loss         = criterion(logits, class_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss  += loss.item() * len(labels)
            total_items += len(labels)
            global_step += 1
            bar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            # ── Mid-epoch checkpoint ───────────────────────────────────────
            if save_steps > 0 and global_step % save_steps == 0:
                save_step_checkpoint(
                    save_dir, model, optimizer, scheduler,
                    epoch, global_step, best_val_mae, history
                )

        # After each epoch: validate and save best model
        start_batch  = 0  # only skip batches in the resumed epoch
        train_loss   = total_loss / max(total_items, 1)
        val_loss, val_mae = evaluate_epoch(model, val_loader, criterion, device, approach)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
        history.append({"epoch": epoch, "global_step": global_step,
                        "train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  ✓ New best model saved (MAE: {best_val_mae:.4f})")

        # Save checkpoint after every epoch too
        save_step_checkpoint(save_dir, model, optimizer, scheduler,
                             epoch + 1, global_step, best_val_mae, history)

    # ── Save training config + completion marker ──────────────────────────
    config = {
        "approach"      : approach,
        "model_name"    : MODEL_NAME,
        "best_val_mae"  : best_val_mae,
        "epochs_run"    : n_epochs,
        "batch_size"    : BATCH_SIZE,
        "learning_rate" : LEARNING_RATE,
        "quick_mode"    : quick_mode,
        "history"       : history,
    }
    with open(save_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Mark training as fully complete (so re-running doesn't restart)
    if not quick_mode:
        with open(complete_marker, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"  Training complete marker saved -> {complete_marker}")

    print(f"\nTraining complete! Best Val MAE: {best_val_mae:.4f}")
    print(f"  Model saved -> {save_dir / 'best_model.pt'}")
    return best_val_mae



def main():
    parser = argparse.ArgumentParser(description="Train Korean movie review score predictor")
    parser.add_argument(
        "--mode", choices=["quick", "full"], default="quick",
        help="quick: 1k samples, 1 epoch to verify pipeline | full: train on all data"
    )
    parser.add_argument(
        "--approach", choices=["regression", "ordinal", "both"], default="both",
        help="Which model approach to train"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Save a resumable checkpoint every N batches (default: 500). Set 0 to disable."
    )
    args = parser.parse_args()

    quick_mode = (args.mode == "quick")

    # Verify datasets exist
    for fname in ["train.pt", "val.pt"]:
        if not (DATA_DIR / fname).exists():
            print(f"❌ ERROR: {DATA_DIR / fname} not found.")
            print("   Please run prepare_data.py first.")
            return

    results = {}

    if args.approach in ("regression", "both"):
        mae = train_model("regression", quick_mode, save_steps=args.save_steps)
        results["regression"] = mae

    if args.approach in ("ordinal", "both"):
        mae = train_model("ordinal", quick_mode, save_steps=args.save_steps)
        results["ordinal"] = mae

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for approach, mae in results.items():
        print(f"  {approach.capitalize():15s} — Best Val MAE: {mae:.4f}")

    if quick_mode:
        print(f"\n⚠  This was a QUICK run ({QUICK_N:,} samples, {QUICK_EPOCHS} epoch). MAE will be high.")
        print("   Run 'python train_model.py --mode full' for real training.")
    print("\nNext step → run: python evaluate_model.py")


if __name__ == "__main__":
    main()
