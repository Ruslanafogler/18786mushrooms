import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

from mushroomCNN import MushroomCNN
from mushroomVIT import MushroomVIT


class _DatasetWithTransform(torch.utils.data.Dataset):
  def __init__(self, base_dataset, indices, transform):
    self.base = base_dataset
    self.indices = indices
    self.transform = transform

  def __len__(self):
    return len(self.indices)
  
  def __getitem__(self, idx):
    #apply transform and return label too
    img, label = self.base.imgs[self.indices[idx]]
    img = Image.open(img).convert("RGB")
    return self.transform(img), label



############################################################################
############################################################################
      
def build_dataloaders(
    data_dir: str,
    img_size: int = 227,
    batch_size: int = 32,
    val_split: float = 0.15, #15% for validation
    test_split: float = 0.15, #15% for test
    num_workers: int = 4,
    augment: bool = True,
    seed: int = 42,
):
  
  '''
  randomly flip/jitter images to make CNN more robust hopefully
  '''

  train_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    *(
      [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3,
                              contrast=0.3,
                              saturation=0.2,
                              hue=0.1),
      ]
      if augment else []
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], #normalizing by mean/std, these are actually acquired from ImageNet averages
                          [0.229, 0.224, 0.225]),
  ])

  eval_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], #normalizing, these are acquired from ImageNet
                          [0.229, 0.224, 0.225]),
  ])

  full_dataset = datasets.ImageFolder(data_dir)
  class_names = full_dataset.classes

  n = len(full_dataset)

  n_test = int(n * test_split)
  n_val = int(n * val_split)
  n_train = n - n_val - n_test #remaining for all the training

  gen = torch.Generator().manual_seed(seed)
  train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [n_train, n_val, n_test],
    generator=gen
  )

  #shuffle and apply class to handle transforms/get item
  train_dataset_tf = _DatasetWithTransform(full_dataset, train_dataset.indices, train_tf)
  val_dataset_tf = _DatasetWithTransform(full_dataset,   val_dataset.indices,   eval_tf)
  test_dataset_tf = _DatasetWithTransform(full_dataset,  test_dataset.indices,  eval_tf)

  loader_params = dict(batch_size=batch_size, 
                        num_workers=num_workers, 
                        pin_memory=True)
  train_loader = DataLoader(train_dataset_tf, shuffle=True,  **loader_params)
  val_loader = DataLoader(val_dataset_tf,     shuffle=False, **loader_params)
  test_loader = DataLoader(test_dataset_tf,   shuffle=False, **loader_params)

  print(f"Dataset splits  →  train: {len(train_dataset_tf)}  "
        f"val: {len(val_dataset_tf)}  test: {len(test_dataset_tf)}")
  print(f"Classes: {class_names}")
  return train_loader, val_loader, test_loader, class_names   


############################################################################
############################################################################

def run_epoch(model, loader, criterion, optimizer, device, phase="train"):
  is_train = (phase == "train")
  model.train() if is_train else model.eval()

  total_loss, correct, total = 0.0, 0, 0
  with torch.set_grad_enabled(is_train):
    for imgs, labels in loader:
      imgs, labels = imgs.to(device), labels.to(device) #send to GPU (potentially)
      logits = model(imgs) #forward call
      loss = criterion(logits, labels) #get the loss

      if is_train:
        optimizer.zero_grad()
        loss.backward() #auto diff the gradients
        optimizer.step()

      total_loss += loss.item() * imgs.size(0) #per pixel?
      preds = logits.argmax(dim=1)
      correct += (preds == labels).sum().item()
      total += imgs.size(0)

    
  return total_loss / total, correct / total


############################################################################
############################################################################
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_labels), np.array(all_preds)


def plot_training_curves(history: dict, title: str, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
 
    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", ms=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   marker="o", ms=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-Entropy Loss per Epoch")
    axes[0].legend(); axes[0].grid(True)
 
    axes[1].plot(epochs, history["train_acc"], label="Train", marker="o", ms=3)
    axes[1].plot(epochs, history["val_acc"],   label="Val",   marker="o", ms=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].legend(); axes[1].grid(True)
 
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved training curves → {save_path}")


############################################################################
############################################################################

def plot_conf_matrix(y_true, y_pred, class_names, title, save_path):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {save_path}")
    
############################################################################
############################################################################


def metrics_summary(y_true, y_pred, class_names, poisonous_idx, split_name):
    print(f"\n{'─'*50}")
    print(f"  {split_name} Results")
    print(f"{'─'*50}")
    acc = (y_true == y_pred).mean()
    print(f"  Accuracy : {acc:.4%}")
    print(classification_report(y_true, y_pred,
                                target_names=class_names, digits=4))
    cm  = confusion_matrix(y_true, y_pred)
    tp  = cm[poisonous_idx, poisonous_idx]
    fn  = cm[poisonous_idx, :].sum() - tp
    fnr = fn / (tp + fn) if (tp + fn) > 0 else float("nan")
    print(f"  False Negative Rate (FNR / miss-poisonous): {fnr:.4%}")


############################################################################
############################################################################

def train_model(
    model,
    model_tag: str,          # "CNN" or "ViT" — used for filenames and print headers
    train_loader,
    val_loader,
    test_loader,
    class_names,
    poisonous_idx: int,
    device,
    out: Path,
    args,
    use_warmup: bool = False,  
):
    # Loss: upweight poisonous class to penalise false negatives
    class_weights = torch.ones(len(class_names), device=device)
    class_weights[poisonous_idx] = 2.0
    criterion = nn.CrossEntropyLoss(weight=class_weights)
 
    optimizer = optim.AdamW(model.parameters(),
                             lr=args.lr, weight_decay=args.weight_decay)
 
    if use_warmup:
        # Linear warmup and cosine decay (shoudl be better for vit)
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            progress = (epoch - args.warmup_epochs) / max(
                1, args.epochs - args.warmup_epochs
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # Pure cosine decay (good default for CNN)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
 
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss    = float("inf")
    patience_counter = 0
    best_ckpt        = out / f"best_{model_tag.lower()}.pt"
 
    print(f"\n{'═'*65}")
    print(f"  Training {model_tag}")
    print(f"{'═'*65}")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  "
          f"{'Train Acc':>9}  {'Val Acc':>8}  {'Time':>6}")
    print("─" * 65)
 
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, device, "train")
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion,
                                    optimizer, device, "val")
        scheduler.step()
 
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
 
        elapsed = time.time() - t0
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {vl_loss:>9.4f}  "
              f"{tr_acc:>9.4%}  {vl_acc:>8.4%}  {elapsed:>5.1f}s")
 
        if vl_loss < best_val_loss:
            best_val_loss    = vl_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no val-loss improvement for {args.patience} epochs)")
                break
 
    # Load best weights
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"Loaded best checkpoint from {best_ckpt}")
 
    # plots 
    tag = model_tag.lower()
    plot_training_curves(
        history,
        title=f"Training Curves — {model_tag}",
        save_path=str(out / f"{tag}_training_curves.png"),
    )
 
    val_true,  val_pred  = collect_predictions(model, val_loader,  device)
    test_true, test_pred = collect_predictions(model, test_loader, device)
 
    #confusion matrices
    plot_conf_matrix(val_true,  val_pred,  class_names,
                     f"{model_tag} — Validation Confusion Matrix",
                     str(out / f"{tag}_val_confusion.png"))
    plot_conf_matrix(test_true, test_pred, class_names,
                     f"{model_tag} — Test Confusion Matrix",
                     str(out / f"{tag}_test_confusion.png"))
 
    # final metrics
    metrics_summary(val_true,  val_pred,  class_names, poisonous_idx, f"{model_tag} Validation")
    metrics_summary(test_true, test_pred, class_names, poisonous_idx, f"{model_tag} Test")
 
    return history
 
 

def parse_args():
    p = argparse.ArgumentParser(
        description="Mushroom Toxicity Classifier — CNN and/or ViT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    # ── which model(s) to run ────────────────────────────────────────────────
    p.add_argument("--model", choices=["cnn", "vit", "both"], default="both",
                   help="Which model to train: cnn, vit, or both")
 
    # ── paths ────────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",   required=True,
                   help="Root folder with 'Edible/' and 'Poisonous/' subfolders")
    p.add_argument("--output_dir", default="./mushroom_output")
 
    # ── data ─────────────────────────────────────────────────────────────────
    p.add_argument("--img_size",    type=int,   default=224,
                   help="Resize images to this square size (must be divisible by --patch_size for ViT)")
    p.add_argument("--val_split",   type=float, default=0.15)
    p.add_argument("--test_split",  type=float, default=0.15)
    p.add_argument("--augment",     action="store_true", default=True)
    p.add_argument("--no_augment",  dest="augment", action="store_false")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)
 
    # ── shared training ───────────────────────────────────────────────────────
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--patience",      type=int,   default=10,
                   help="Early-stopping patience (epochs without val-loss improvement)")
    p.add_argument("--warmup_epochs", type=int,   default=5,
                   help="Linear LR warmup epochs (ViT only)")
 
    # ── CNN-specific ──────────────────────────────────────────────────────────
    cnn = p.add_argument_group("CNN hyperparameters")
    cnn.add_argument("--num_conv_blocks", type=int,   default=3)
    cnn.add_argument("--base_filters",    type=int,   default=32)
    cnn.add_argument("--kernel_size",     type=int,   default=3)
    cnn.add_argument("--stride",          type=int,   default=1)
    cnn.add_argument("--padding",         type=int,   default=1)
    cnn.add_argument("--pool_every",      type=int,   default=1,
                     help="Insert MaxPool after every N conv blocks (0=never)")
    cnn.add_argument("--pool_kernel",     type=int,   default=2)
    cnn.add_argument("--conv_dropout",    type=float, default=0.0)
    cnn.add_argument("--depthwise",       action="store_true", default=False)
    cnn.add_argument("--fc_hidden",       type=int,   default=256)
    cnn.add_argument("--fc_dropout",      type=float, default=0.5)
 
    # ── ViT-specific ──────────────────────────────────────────────────────────
    vit = p.add_argument_group("ViT hyperparameters")
    vit.add_argument("--patch_size",   type=int,   default=16,
                     help="Patch size P; img_size must be divisible by P")
    vit.add_argument("--embed_dim",    type=int,   default=256,
                     help="Token embedding dimension; must be divisible by num_heads")
    vit.add_argument("--num_heads",    type=int,   default=8)
    vit.add_argument("--depth",        type=int,   default=6,
                     help="Number of Transformer encoder blocks")
    vit.add_argument("--mlp_ratio",    type=float, default=4.0)
    vit.add_argument("--attn_dropout", type=float, default=0.0)
    vit.add_argument("--mlp_dropout",  type=float, default=0.1)
    vit.add_argument("--head_dropout", type=float, default=0.5)
 
    return p.parse_args()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    args = parse_args()
 
    # validate ViT constraints before doing anything
    if args.model in ("vit", "both"):
        assert args.img_size % args.patch_size == 0, (
            f"--img_size {args.img_size} must be divisible by --patch_size {args.patch_size}"
        )
        assert args.embed_dim % args.num_heads == 0, (
            f"--embed_dim {args.embed_dim} must be divisible by --num_heads {args.num_heads}"
        )
 
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
 
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device : {device}")
    print(f"Running model: {args.model}")
 
    # ── shared data loaders ──────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        augment=args.augment,
        seed=args.seed,
    )
 
    poisonous_idx = next(
        (i for i, c in enumerate(class_names) if c.lower() == "poisonous"), 1
    )
    print(f"Positive class (poisonous) index: {poisonous_idx}")
 
    # ── CNN ──────────────────────────────────────────────────────────────────
    if args.model in ("cnn", "both"):
        cnn = MushroomCNN(
            num_conv_blocks=args.num_conv_blocks,
            base_filters=args.base_filters,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            pool_every=args.pool_every,
            pool_kernel=args.pool_kernel,
            conv_dropout=args.conv_dropout,
            depthwise=args.depthwise,
            fc_hidden=args.fc_hidden,
            fc_dropout=args.fc_dropout,
            num_classes=len(class_names),
        ).to(device)
 
        print(f"\nCNN parameters: {sum(p.numel() for p in cnn.parameters() if p.requires_grad):,}")
        print(cnn)
 
        train_model(
            model=cnn,
            model_tag="CNN",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            poisonous_idx=poisonous_idx,
            device=device,
            out=out,
            args=args,
            use_warmup=False,
        )
 
    # ── ViT ──────────────────────────────────────────────────────────────────
    if args.model in ("vit", "both"):
        vit = MushroomVIT(
            img_size=args.img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            depth=args.depth,
            mlp_ratio=args.mlp_ratio,
            attn_dropout=args.attn_dropout,
            mlp_dropout=args.mlp_dropout,
            head_dropout=args.head_dropout,
            num_classes=len(class_names),
        ).to(device)
 
        num_patches = vit.patch_embed.num_patches
        grid        = int(math.sqrt(num_patches))
        print(f"\nViT patch grid  : {grid}×{grid} = {num_patches} patches "
              f"of size {args.patch_size}×{args.patch_size}")
        print(f"Sequence length : {num_patches + 1} (patches + CLS token)")
        print(f"ViT parameters  : {sum(p.numel() for p in vit.parameters() if p.requires_grad):,}")
        print(vit)
 
        train_model(
            model=vit,
            model_tag="ViT",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            poisonous_idx=poisonous_idx,
            device=device,
            out=out,
            args=args,
            use_warmup=True,
        )
 
    print(f"\nAll outputs saved to: {out.resolve()}")
 
 
if __name__ == "__main__":
    main()