# here is our sweep for the CNN, ViT, and ResNet mushroom stuff
# usage ex:!!!
#   python mushroom_sweep.py --data_dir ./data --output_dir ./sweep_out
#   python mushroom_sweep.py --data_dir ./data --arch vit --epochs 15
#   python mushroom_sweep.py --data_dir ./data --configs vit_baseline vit_less_reg

import argparse
import csv
import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

from mushroomMain import (
    train_model,
    build_dataloaders,
    collect_predictions,
)

from mushroom_explain import (
    GradCAM,
    vit_forward_with_attn,
    attention_rollout,
    overlay_heatmap,
    softmax,
)

from mushroomCNN import MushroomCNN
from mushroomVIT import MushroomVIT
from mushroomResNet import MushroomResNet



CNN_CONFIGS = [
    dict(
        name="cnn_baseline",
        arch="cnn",
        model_kwargs=dict(),
        train_overrides=dict(),
    ),
    dict(
        name="cnn_heavier_reg",
        arch="cnn",
        model_kwargs=dict(conv_dropout=0.1, fc_dropout=0.6),
        train_overrides=dict(),
    ),
    dict(
        name="cnn_deeper",
        arch="cnn",
        model_kwargs=dict(num_conv_blocks=4),
        train_overrides=dict(),
    ),
    dict(
        name="cnn_wider_kernel",
        arch="cnn",
        # kernel=5 needs padding=2 to preserve spatial size at stride 1
        model_kwargs=dict(kernel_size=5, padding=2),
        train_overrides=dict(),
    ),
]

VIT_CONFIGS = [
    dict(
        #default parameter version
        name="vit_baseline",
        arch="vit",
        model_kwargs=dict(),
        train_overrides=dict(),
    ),
    dict(
        name="vit_smaller",
        #less heads and depth, smaller than baseline 
        arch="vit",
        model_kwargs=dict(embed_dim=128, num_heads=4, depth=4),
        train_overrides=dict(),
    ),
    dict(
        name="vit_less_reg",
        #less regularization versoin
        arch="vit",
        model_kwargs=dict(mlp_dropout=0.0, head_dropout=0.2),
        train_overrides=dict(),
    ),
    dict(
        #MORE depth version
        name="vit_deeper",
        arch="vit",
        model_kwargs=dict(depth=8),
        train_overrides=dict(),
    ),
]

ALL_CONFIGS = CNN_CONFIGS + VIT_CONFIGS

RESNET_CONFIGS = [
    dict(
        name="resnet_baseline",
        arch="resnet",
        model_kwargs=dict(preset="resnet18"),
        train_overrides=dict(),
    ),
    dict(
        name="resnet_light_reg",
        #less regularization (dropout) just to see
        arch="resnet",
        model_kwargs=dict(preset="resnet18", head_dropout=0.3),
        train_overrides=dict(),
    ),
    dict(
        name="resnet_block_drop",
        #MORE dropout but in the residual blocks :0 
        arch="resnet",
        model_kwargs=dict(preset="resnet18", block_dropout=0.1),
        train_overrides=dict(),
    ),
    dict(
        name="resnet_narrow",
        #half the channel widths (reduce parameters), see if narrower has the same or less effect to check if our baseline is overkill
        arch="resnet",
        model_kwargs=dict(preset="resnet18", base_width=32),
        train_overrides=dict(),
    ),
    dict(
        name="resnet34",
        #deeper  
        arch="resnet",
        model_kwargs=dict(preset="resnet34"),
        train_overrides=dict(),
    ),
]

ALL_CONFIGS = CNN_CONFIGS + VIT_CONFIGS + RESNET_CONFIGS



def pick_fixed_samples(dataset, test_indices, num_samples, class_names, seed):
    """Balanced edible/poisonous sample of the test split, seeded for
    reproducibility.  The same indices are used across every config so that
    heatmaps can be compared row-wise in the final montage."""
    poison_idx = next((i for i, c in enumerate(class_names)
                       if c.lower() == "poisonous"), 1)
    edible_idx = next((i for i, c in enumerate(class_names)
                       if c.lower() == "edible"),
                      0 if poison_idx != 0 else 1)

    rng = np.random.default_rng(seed)
    pois = [i for i in test_indices if dataset.imgs[i][1] == poison_idx]
    edi  = [i for i in test_indices if dataset.imgs[i][1] == edible_idx]
    pois = [pois[i] for i in rng.permutation(len(pois))]
    edi  = [edi[i]  for i in rng.permutation(len(edi))]

    n_pois = num_samples // 2
    n_edi  = num_samples - n_pois
    chosen = pois[:n_pois] + edi[:n_edi]
    return chosen


def heatmap_for_sample(model, arch, x, gradcam=None):
    if arch in ("cnn", "resnet"):
        cam, logits, pred = gradcam(x)
        return cam, logits, pred
    else:
        logits_t, attentions = vit_forward_with_attn(model, x)
        logits = logits_t.detach().cpu().numpy()[0]
        pred = int(np.argmax(logits))
        roll = attention_rollout(attentions, head_fusion="mean", discard_ratio=0.1)
        return roll, logits, pred


def compute_heatmaps_for_config(model, arch, fixed_samples, dataset,
                                 transform, device, img_size):
    gradcam = GradCAM(model, model.features) if arch in ("cnn", "resnet") else None
    out = []
    try:
        for idx in fixed_samples:
            path, label = dataset.imgs[idx]
            img_pil = Image.open(path).convert("RGB")
            img_resized = img_pil.resize((img_size, img_size))
            img_arr = np.array(img_resized).astype(np.float32) / 255.0
            x = transform(img_pil).unsqueeze(0).to(device)

            heat, logits, pred = heatmap_for_sample(model, arch, x, gradcam)
            out.append(dict(
                idx=idx, label=label, img_arr=img_arr,
                heatmap=heat, logits=logits, pred=pred,
            ))
    finally:
        if gradcam is not None:
            gradcam.remove()
    return out


def _title_colour(pred_name, true_name):
    danger = pred_name.lower() == "edible" and true_name.lower() == "poisonous"
    if danger:   return "red",    "✗ DANGER"
    if pred_name == true_name: return "green",  "✓"
    return "orange", "✗"


def plot_per_config_explanation(heatmaps, class_names, title, out_path, img_size):
    n = len(heatmaps)
    fig, axes = plt.subplots(n, 2, figsize=(7.5, 3.4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for row, h in enumerate(heatmaps):
        true_name = class_names[h["label"]]
        pred_name = class_names[h["pred"]]
        probs = softmax(h["logits"])
        overlay = overlay_heatmap(h["img_arr"], h["heatmap"], size=img_size)
        colour, marker = _title_colour(pred_name, true_name)
        prob_str = "  ".join(f"{c}: {p:.2f}" for c, p in zip(class_names, probs))

        axes[row, 0].imshow(h["img_arr"])
        axes[row, 0].set_title(f"true: {true_name}", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(
            f"{marker}  pred: {pred_name}\n{prob_str}",
            fontsize=9, color=colour,
        )
        axes[row, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def plot_comparison_montage(arch_label, config_heatmaps, class_names,
                             img_size, out_path):
    config_names = list(config_heatmaps.keys())
    if not config_names:
        return
    n_samples = len(next(iter(config_heatmaps.values())))
    n_cols = 1 + len(config_names)

    fig, axes = plt.subplots(n_samples, n_cols,
                              figsize=(2.7 * n_cols, 3.0 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"{arch_label} — same samples across every config",
        fontsize=13, fontweight="bold",
    )

    first_cfg = config_heatmaps[config_names[0]]

    for row in range(n_samples):
        ref = first_cfg[row]
        true_name = class_names[ref["label"]]

        axes[row, 0].imshow(ref["img_arr"])
        axes[row, 0].set_title(f"original\ntrue: {true_name}", fontsize=9)
        axes[row, 0].axis("off")

        for col, cname in enumerate(config_names, start=1):
            h = config_heatmaps[cname][row]
            pred_name = class_names[h["pred"]]
            probs = softmax(h["logits"])
            p_pred = probs[h["pred"]]
            overlay = overlay_heatmap(h["img_arr"], h["heatmap"], size=img_size)
            colour, marker = _title_colour(pred_name, true_name)

            axes[row, col].imshow(overlay)
            # Strip the arch prefix so the column title fits
            short = cname.split("_", 1)[1] if "_" in cname else cname
            axes[row, col].set_title(
                f"{short}\n{marker} {pred_name} ({p_pred:.2f})",
                fontsize=8, color=colour,
            )
            axes[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")

#make a bar chart for this
def plot_sweep_summary(rows, arch_filter, out_path):
    grows = [r for r in rows if r["arch"] == arch_filter]
    if not grows:
        return
    grows = sorted(grows, key=lambda r: r["poisonous_fnr"])

    names = [r["name"].split("_", 1)[1] if "_" in r["name"] else r["name"]
             for r in grows]
    acc = [r["test_accuracy"] * 100 for r in grows]
    fnr = [r["poisonous_fnr"]  * 100 for r in grows]

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.4 * len(names) + 3), 4.5))

    b0 = axes[0].bar(names, acc, color="#3b6ea5")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title(f"{arch_filter.upper()} sweep — accuracy")
    axes[0].set_ylim(0, 100); axes[0].grid(True, axis="y", alpha=0.3)
    for b, v in zip(b0, acc):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=9)

    b1 = axes[1].bar(names, fnr, color="#c44e52")
    axes[1].set_ylabel("Poisonous FNR (%)  ← lower is safer")
    axes[1].set_title(f"{arch_filter.upper()} sweep — false negative rate")
    axes[1].set_ylim(0, max(20, max(fnr) * 1.3) if fnr else 20)
    axes[1].grid(True, axis="y", alpha=0.3)
    for b, v in zip(b1, fnr):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=9)

    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path}")


def build_model(cfg, num_classes, img_size):
    if cfg["arch"] == "cnn":
        return MushroomCNN(num_classes=num_classes, **cfg["model_kwargs"])
    elif cfg["arch"] == "resnet":
        return MushroomResNet(num_classes=num_classes, **cfg["model_kwargs"])
    else:
        return MushroomVIT(img_size=img_size,
                            num_classes=num_classes,
                            **cfg["model_kwargs"])


def run_config(cfg, base_args, train_loader, val_loader, test_loader,
               class_names, poisonous_idx, device, fixed_samples, dataset,
               transform, out_root):

    print(f"\n{'═'*70}")
    print(f"  Config: {cfg['name']}")
    print(f"  model_kwargs={cfg['model_kwargs']}  "
          f"train_overrides={cfg['train_overrides']}")
    print(f"{'═'*70}")

    cfg_dir = out_root / cfg["name"]
    cfg_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, len(class_names), base_args.img_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}")

    # Per-config args: copy base args, then apply overrides for this run
    args_cfg = Namespace(**vars(base_args))
    for k, v in cfg["train_overrides"].items():
        setattr(args_cfg, k, v)

    tag = {"cnn": "CNN", "vit": "ViT", "resnet": "ResNet"}[cfg["arch"]]
    use_warmup = (cfg["arch"] == "vit")

    t0 = time.time()
    train_model(
        model=model,
        model_tag=tag,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        poisonous_idx=poisonous_idx,
        device=device,
        out=cfg_dir,
        args=args_cfg,
        use_warmup=use_warmup,
    )
    elapsed = time.time() - t0

    ckpt_path = cfg_dir / f"best_{tag.lower()}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

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

    heatmaps = compute_heatmaps_for_config(
        model, cfg["arch"], fixed_samples, dataset,
        transform, device, base_args.img_size,
    )
    plot_per_config_explanation(
        heatmaps, class_names,
        title=f"{cfg['name']}  ({tag})  —  acc {acc:.2%}  FNR {fnr:.2%}",
        out_path=cfg_dir / "explanations.png",
        img_size=base_args.img_size,
    )

    #Free the model 
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    row = dict(
        name=cfg["name"],
        arch=cfg["arch"],
        params=n_params,
        model_kwargs=json.dumps(cfg["model_kwargs"]),
        train_overrides=json.dumps(cfg["train_overrides"]),
        test_accuracy=acc,
        poisonous_precision=precision,
        poisonous_recall=recall,
        poisonous_fnr=fnr,
        tp=tp, fp=fp, tn=tn, fn=fn,
        elapsed_sec=round(elapsed, 1),
    )

    print(f"\n  → acc {acc:.3%}  recall {recall:.3%}  FNR {fnr:.3%}  "
          f"({elapsed:.0f}s)")

    return row, heatmaps



def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter sweep for mushroom CNN + ViT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--output_dir", default="./sweep_out")
    p.add_argument("--arch", choices=["cnn", "vit", "resnet", "both", "all"],
                   default="all",
                   help="Which architectures to sweep (both=cnn+vit, all=cnn+vit+resnet)")
    p.add_argument("--configs", nargs="+", default=None,
                   help="Run only the named configs (default: all for --arch)")

    p.add_argument("--epochs",        type=int,   default=12)
    p.add_argument("--patience",      type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--warmup_epochs", type=int,   default=3)
    p.add_argument("--img_size",      type=int,   default=224)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42)

    #explainability
    p.add_argument("--num_explain_samples", type=int, default=4,
                   help="Number of fixed test samples to explain per config")
    p.add_argument("--sample_seed", type=int, default=123,
                   help="Seed controlling which test samples are explained")
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

    # Which configs to run
    pool = ALL_CONFIGS
    if args.arch == "cnn":
        pool = CNN_CONFIGS
    elif args.arch == "vit":
        pool = VIT_CONFIGS
    elif args.arch == "resnet":
        pool = RESNET_CONFIGS
    elif args.arch == "both":
        pool = CNN_CONFIGS + VIT_CONFIGS
    if args.configs:
        wanted = set(args.configs)
        pool = [c for c in pool if c["name"] in wanted]
        missing = wanted - {c["name"] for c in pool}
        if missing:
            known = ", ".join(c["name"] for c in ALL_CONFIGS)
            raise SystemExit(f"Unknown configs: {missing}.  Known: {known}")
    if not pool:
        raise SystemExit("Nothing to run.")

    print(f"Will run {len(pool)} config(s): {[c['name'] for c in pool]}\n")

    #shared dataloaders so the split and augments are the same 
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=0.15,
        test_split=0.15,
        num_workers=args.num_workers,
        augment=True,
        seed=args.seed,
    )
    poisonous_idx = next(
        (i for i, c in enumerate(class_names) if c.lower() == "poisonous"), 1
    )

    full_dataset = datasets.ImageFolder(args.data_dir)
    n = len(full_dataset)
    n_test = int(n * 0.15)
    n_val  = int(n * 0.15)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(args.seed)
    _, _, test_split = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test], generator=gen
    )
    test_indices = list(test_split.indices)

    fixed_samples = pick_fixed_samples(
        full_dataset, test_indices, args.num_explain_samples,
        class_names, args.sample_seed,
    )
    print(f"Fixed explain samples: {fixed_samples}")

    explain_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows = []
    heatmaps_by_config = {"cnn": {}, "vit": {}, "resnet": {}}

    for i, cfg in enumerate(pool, 1):
        print(f"\n[{i}/{len(pool)}] ", end="")
        row, heatmaps = run_config(
            cfg, args,
            train_loader, val_loader, test_loader,
            class_names, poisonous_idx, device,
            fixed_samples, full_dataset, explain_tf, out,
        )
        rows.append(row)
        heatmaps_by_config[cfg["arch"]][cfg["name"]] = heatmaps

    print(f"\n{'═'*70}")
    print("  Sweep complete — writing summary outputs")
    print(f"{'═'*70}")

    sorted_rows = sorted(
        rows,
        key=lambda r: (r["arch"], r["poisonous_fnr"], -r["test_accuracy"]),
    )

    fields = [
        "name", "arch", "params", "test_accuracy",
        "poisonous_precision", "poisonous_recall", "poisonous_fnr",
        "tp", "fp", "tn", "fn",
        "model_kwargs", "train_overrides", "elapsed_sec",
    ]
    with open(out / "sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    with open(out / "sweep_results.json", "w") as f:
        json.dump(sorted_rows, f, indent=2)
    print(f"  saved → {out / 'sweep_results.csv'}")
    print(f"  saved → {out / 'sweep_results.json'}")

    for arch_key, arch_label, explain_label in [
        ("cnn",    "cnn",    "CNN — Grad-CAM"),
        ("vit",    "vit",    "ViT — attention rollout"),
        ("resnet", "resnet", "ResNet — Grad-CAM"),
    ]:
        if any(r["arch"] == arch_key for r in rows):
            plot_sweep_summary(rows, arch_key,
                               out / f"sweep_summary_{arch_key}.png")
            plot_comparison_montage(
                explain_label,
                heatmaps_by_config[arch_key], class_names,
                args.img_size,
                out / f"comparison_explanations_{arch_key}.png",
            )

    print("\nLeaderboard (sorted by FNR asc, then accuracy desc):")
    print(f"  {'config':<22} {'arch':>5} {'acc':>8} {'recall':>8} {'FNR':>7} "
          f"{'params':>10}")
    print(f"  {'-'*22} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*10}")
    for r in sorted_rows:
        print(f"  {r['name']:<22} {r['arch']:>5} "
              f"{r['test_accuracy']:>8.3%} "
              f"{r['poisonous_recall']:>8.3%} "
              f"{r['poisonous_fnr']:>7.3%} "
              f"{r['params']:>10,}")

    print(f"\nAll outputs saved under {out.resolve()}")


if __name__ == "__main__":
    main()
