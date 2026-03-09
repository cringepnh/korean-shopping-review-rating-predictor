"""
evaluate_model.py
=================
STEP 5: Evaluate the trained models on the held-out test set.

Metrics reported:
  - MAE  (Mean Absolute Error) — average score distance from truth
  - RMSE (Root Mean Squared Error) — penalizes larger errors more
  - Accuracy within 1 point — % of predictions within ±1 of true score

Outputs:
  - Printed metrics table comparing regression vs ordinal
  - data/confusion_matrix_regression.png
  - data/confusion_matrix_ordinal.png

Run:
  python evaluate_model.py
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
tqdm.monitor_interval = 0
from train_model import (
    KoELECTRARegressor,
    KoELECTRAOrdinalClassifier,
    ReviewDataset,
    MODEL_NAME,
    BATCH_SIZE,
    NUM_CLASSES,
)

DATA_DIR   = Path("data")
MODELS_DIR = Path("models")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_all(model, loader, approach, device):
    """
    Run model on entire DataLoader, return:
      preds  : list of predicted scores (1-10)
      labels : list of true scores (1-10)
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc  = "  🔴 Predicting",
            unit  = "batch",
            colour= "red",
            dynamic_ncols=True,
        ):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].cpu().numpy().tolist()

            if approach == "regression":
                preds = model(input_ids, attention_mask).cpu().numpy()
                # Round to nearest integer for reporting
                preds = np.round(np.clip(preds, 1, 10)).astype(int).tolist()
            else:
                logits = model(input_ids, attention_mask)
                preds  = (torch.argmax(logits, dim=1) + 1).cpu().numpy().tolist()

            all_preds.extend(preds)
            all_labels.extend([int(l) for l in labels])

    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute MAE, RMSE, and accuracy-within-1."""
    mae   = float(np.mean(np.abs(preds - labels)))
    rmse  = float(np.sqrt(np.mean((preds - labels) ** 2)))
    acc1  = float(np.mean(np.abs(preds - labels) <= 1) * 100)  # % within ±1
    return {"MAE": mae, "RMSE": rmse, "Acc±1 (%)": acc1}


def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray,
                          title: str, save_path: Path) -> None:
    """Plot and save a 10×10 confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds, labels=list(range(1, 11)))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=range(1, 11), yticklabels=range(1, 11), ax=ax,
    )
    ax.set_xlabel("Predicted Score", fontsize=12)
    ax.set_ylabel("True Score", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix saved → {save_path}")


def evaluate_approach(approach: str, test_ds, device):
    """Load the best checkpoint for an approach and evaluate on test set."""
    model_dir = MODELS_DIR / approach
    ckpt_path = model_dir / "best_model.pt"

    if not ckpt_path.exists():
        print(f"  ⚠ No checkpoint found at {ckpt_path}. Skipping {approach}.")
        return None

    # Build the same model architecture
    if approach == "regression":
        model = KoELECTRARegressor(MODEL_NAME)
    else:
        model = KoELECTRAOrdinalClassifier(MODEL_NAME, NUM_CLASSES)

    # Load saved weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n  Running predictions for {approach} ...")
    preds, labels = predict_all(model, test_loader, approach, device)

    metrics = compute_metrics(preds, labels)

    # Save confusion matrix
    cm_path = DATA_DIR / f"confusion_matrix_{approach}.png"
    plot_confusion_matrix(
        preds, labels,
        title=f"Confusion Matrix — {approach.capitalize()} Model",
        save_path=cm_path,
    )

    return metrics, preds, labels


def main():
    print("=" * 60)
    print("Step 5: Model Evaluation on Test Set")
    print("=" * 60)

    # Load test dataset
    test_path = DATA_DIR / "test.pt"
    if not test_path.exists():
        print(f"❌ ERROR: {test_path} not found.")
        print("   Please run prepare_data.py first.")
        return

    print("\nLoading test dataset ...")
    test_data = torch.load(test_path, weights_only=True)
    test_ds   = ReviewDataset(test_data)
    print(f"  Test set size: {len(test_ds):,} samples")

    device = get_device()
    results = {}

    # Evaluate both approaches
    for approach in ["regression", "ordinal"]:
        print(f"\n{'─'*50}")
        print(f"Evaluating: {approach.upper()}")
        print(f"{'─'*50}")
        result = evaluate_approach(approach, test_ds, device)
        if result is not None:
            metrics, preds, labels = result
            results[approach] = metrics

            print(f"\n  Results for {approach.upper()} model:")
            print(f"    MAE           : {metrics['MAE']:.4f}  (lower is better)")
            print(f"    RMSE          : {metrics['RMSE']:.4f}  (lower is better)")
            print(f"    Accuracy ±1   : {metrics['Acc±1 (%)']:.2f}%  (higher is better)")

            print(f"\n  Score-by-score breakdown:")
            for s in range(1, 11):
                mask  = labels == s
                n     = mask.sum()
                if n == 0:
                    continue
                s_mae = np.mean(np.abs(preds[mask] - labels[mask]))
                s_acc = 100 * np.mean(np.abs(preds[mask] - labels[mask]) <= 1)
                print(f"    Score {s:2d} (n={n:5,}): MAE={s_mae:.3f}, Acc±1={s_acc:.1f}%")

    # ── Side-by-side comparison ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL COMPARISON: REGRESSION vs ORDINAL")
    print("="*60)
    print(f"\n  {'Metric':<20} {'Regression':>15} {'Ordinal':>15} {'Winner':>12}")
    print("  " + "-"*62)

    for metric in ["MAE", "RMSE", "Acc±1 (%)"]:
        reg_val = results.get("regression", {}).get(metric, float("nan"))
        ord_val = results.get("ordinal",    {}).get(metric, float("nan"))

        if metric.startswith("Acc"):
            winner = "Regression" if reg_val > ord_val else "Ordinal"
            better = "↑"
        else:
            winner = "Regression" if reg_val < ord_val else "Ordinal"
            better = "↓"

        print(f"  {metric:<20} {reg_val:>14.4f} {ord_val:>14.4f}  {winner:>10} {better}")

    print(f"\n  Interpretation:")
    print(f"    MAE ≤ 1.5 is generally good for a 10-point scale")
    print(f"    Acc±1 ≥ 60% means most predictions are within 1 star of truth")
    print("\n  Confusion matrices saved in: data/")
    print("\nNext step → run: python predict.py")


if __name__ == "__main__":
    main()
