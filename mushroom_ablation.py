"""
mushroom_ablation.py
---------------------
Runs the planned ablation experiments from the milestone:

  1. Sensitivity to the poisonous class weight  (1.0, 1.5, 2.0, 3.0, 5.0)
  2. Effect of data augmentation                (on vs off)
  3. Model capacity                             (depth + width sweep)

For each experiment we train a fresh CNN from the same seed/split and
record test-set metrics, with emphasis on poisonous false negative rate.
Results are written to a CSV and rendered as comparison bar charts.

Example:
  python mushroom_ablation.py --data_dir ./data --output_dir ./ablation_out
  python mushroom_ablation.py --data_dir ./data --epochs 8 --groups weight capacity
"""

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from mushroomCNN import MushroomCNN
from mushroomMain import build_dataloaders, run_epoch, collect_predictions


# ─────────────────────────────────────────────────────────────────────────────
# Experiment definitions
# ─────────────────────────────────────────────────────────────────────────────

# CNN capacity presets.  fc_hidden + base_filters + num_conv_blocks together
# control parameter count; we sweep along several dimensions.
CAPACITY_PRESETS = {
    "tiny":     dict(num_conv_blocks=2, base_filters=16, fc_hidden=64),
    "small":    dict(num_conv_blocks=2, base_filters=32, fc_hidden=128),
    "baseline": dict(num_conv_blocks=3, base_filters=32, fc_hidden=256),
    "deep":     dict(num_conv_blocks=4, base_filters=32, fc_hidden=256),
    "wide":     dict(num_conv_blocks=3, base_filters=64, fc_hidden=512),
}


def build_experiment_list(groups):
    """
    Build the (ordered) list of experiments to run, filtered by `groups`.
    Each experiment is a dict with keys:
      name, group, augment, poisonous_weight, capacity
    """
    exps = []

    if "weight" in groups:
        for w in (1.0, 1.5, 2.0, 3.0, 5.0):
            exps.append(dict(
                name=f"weight_{w}",
                group="class_weight",
                augment=True,
                poisonous_weight=w,
                capacity="baseline",
            ))

    if "augment" in groups:
        for aug in (True, False):
            exps.append(dict(
                name=f"augment_{'on' if aug else 'off'}",
                group="augmentation",
                augment=aug,
                poisonous_weight=2.0,
                capacity="baseline",
            ))

    if "capacity" in groups:
        for cap in ("tiny", "small", "baseline", "deep", "wide"):
            exps.append(dict(
                name=f"capacity_{cap}",
                group="capacity",
                augment=True,
                poisonous_weight=2.0,
                capacity=cap,
            ))

    return exps


# ─────────────────────────────────────────────────────────────────────────────
# Streamlined trainer (parameterised on poisonous_weight)
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    model,
    train_loader, val_loader, test_loader,
    class_names, poisonous_idx, device,
    epochs, lr, weight_decay, poisonous_weight, patience,
    verbose=True,
):
    """
    Train one model with the given poisonous class weight, early-stop on
    val loss, and return test-set metrics from the best checkpoint.
    """
    weights = torch.ones(len(class_names), device=device)
    weights[poisonous_idx] = poisonous_weight
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, "train")
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, "val")
        scheduler.step()

        if verbose:
            print(f"    epoch {epoch:>2}  train {tr_loss:.4f}/{tr_acc:.3%}  "
                  f"val {vl_loss:.4f}/{vl_acc:.3%}  ({time.time()-t0:.1f}s)")

        if vl_loss < best_val:
            best_val = vl_loss
            # cache on cpu so we don't accumulate gpu memory across experiments
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    early stop @ epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # test-set metrics
    y_true, y_pred = collect_predictions(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred)
    acc = float((y_true == y_pred).mean())

    tp = int(cm[poisonous_idx, poisonous_idx])
    fn = int(cm[poisonous_idx, :].sum() - tp)
    fp = int(cm[:, poisonous_idx].sum() - tp)
    tn = int(len(y_true) - tp - fn - fp)

    fnr       = fn / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    return {
        "test_accuracy":        acc,
        "poisonous_precision":  precision,
        "poisonous_recall":     recall,
        "poisonous_fnr":        fnr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "best_val_loss":        best_val,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_group(rows, group_name, out_path):
    """Side-by-side bar chart: accuracy and poisonous FNR per experiment in a group."""
    g_rows = [r for r in rows if r["group"] == group_name]
    if not g_rows:
        return

    names = [r["name"] for r in g_rows]
    acc   = [r["test_accuracy"]  * 100 for r in g_rows]
    fnr   = [r["poisonous_fnr"]  * 100 for r in g_rows]

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.5 * len(names) + 2), 4.5))

    bars0 = axes[0].bar(names, acc, color="#3b6ea5")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title(f"{group_name} — accuracy")
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars0, acc):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=9)

    bars1 = axes[1].bar(names, fnr, color="#c44e52")
    axes[1].set_ylabel("Poisonous FNR (%)  ← lower is safer")
    axes[1].set_title(f"{group_name} — poisonous false negative rate")
    axes[1].set_ylim(0, max(20, max(fnr) * 1.3) if fnr else 20)
    axes[1].grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars1, fnr):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=9)

    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


def write_csv(rows, out_path):
    fields = [
        "name", "group", "capacity", "augment", "poisonous_weight",
        "test_accuracy", "poisonous_precision", "poisonous_recall",
        "poisonous_fnr", "tp", "fp", "tn", "fn",
        "params", "best_val_loss", "elapsed_sec",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"  saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Mushroom CNN ablation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--output_dir", default="./ablation_out")
    p.add_argument("--groups", nargs="+",
                   choices=["weight", "augment", "capacity"],
                   default=["weight", "augment", "capacity"],
                   help="Which ablation groups to run")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--patience",     type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--img_size",     type=int,   default=224)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--quiet",        action="store_true",
                   help="Suppress per-epoch logging")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    experiments = build_experiment_list(args.groups)
    print(f"Running {len(experiments)} experiments across groups: {args.groups}\n")

    # We build dataloaders once per (augment value) since augment changes train_tf.
    # All other ablations reuse the same loaders for an exact apples-to-apples split.
    loader_cache = {}

    def get_loaders(augment):
        if augment not in loader_cache:
            loader_cache[augment] = build_dataloaders(
                data_dir=args.data_dir,
                img_size=args.img_size,
                batch_size=args.batch_size,
                val_split=0.15,
                test_split=0.15,
                num_workers=args.num_workers,
                augment=augment,
                seed=args.seed,
            )
        return loader_cache[augment]

    rows = []

    for i, exp in enumerate(experiments, 1):
        print(f"[{i}/{len(experiments)}] {exp['name']}  "
              f"(weight={exp['poisonous_weight']}, augment={exp['augment']}, "
              f"capacity={exp['capacity']})")

        train_loader, val_loader, test_loader, class_names = get_loaders(exp["augment"])
        poisonous_idx = next(
            (i for i, c in enumerate(class_names) if c.lower() == "poisonous"), 1
        )

        cap_cfg = CAPACITY_PRESETS[exp["capacity"]]
        model = MushroomCNN(
            num_conv_blocks=cap_cfg["num_conv_blocks"],
            base_filters=cap_cfg["base_filters"],
            fc_hidden=cap_cfg["fc_hidden"],
            num_classes=len(class_names),
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params: {n_params:,}")

        t0 = time.time()
        metrics = train_and_evaluate(
            model,
            train_loader, val_loader, test_loader,
            class_names, poisonous_idx, device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            poisonous_weight=exp["poisonous_weight"],
            patience=args.patience,
            verbose=not args.quiet,
        )
        elapsed = time.time() - t0

        row = {
            **exp,
            **metrics,
            "params": n_params,
            "elapsed_sec": round(elapsed, 1),
        }
        rows.append(row)

        print(f"  → acc {metrics['test_accuracy']:.3%}  "
              f"poisonous recall {metrics['poisonous_recall']:.3%}  "
              f"FNR {metrics['poisonous_fnr']:.3%}  "
              f"({elapsed:.0f}s)\n")

        # Free memory for the next experiment
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("=" * 60)
    print("Ablation complete — writing outputs")
    print("=" * 60)

    write_csv(rows, out / "ablation_results.csv")
    with open(out / "ablation_results.json", "w") as f:
        json.dump(rows, f, indent=2)

    for group in {r["group"] for r in rows}:
        plot_group(rows, group, out / f"ablation_{group}.png")

    # A small printed summary table
    print("\nSummary:")
    print(f"  {'name':<22} {'acc':>7} {'recall':>7} {'FNR':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7}")
    for r in rows:
        print(f"  {r['name']:<22} {r['test_accuracy']:>7.3%} "
              f"{r['poisonous_recall']:>7.3%} {r['poisonous_fnr']:>7.3%}")


if __name__ == "__main__":
    main()
