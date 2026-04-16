"""
mushroom_explain.py
-------------------
Explainability analysis for the mushroom classifiers.

Given trained CNN and ViT checkpoints, this script:
  • picks a stratified set of test images (correct & incorrect, edible & poisonous)
  • computes Grad-CAM for the CNN  — gradients × activations at the last conv stage
  • computes attention rollout for the ViT — multi-layer CLS-token saliency
  • saves a side-by-side panel per image:
        [original | CNN Grad-CAM overlay | ViT attention overlay]
    annotated with predicted/true label and softmax probabilities

For a safety-critical classifier, knowing *why* it flagged a mushroom matters
as much as knowing *what* it flagged.

Example:
  python mushroom_explain.py \\
      --data_dir ./data \\
      --cnn_ckpt ./mushroom_output/best_cnn.pt \\
      --vit_ckpt ./mushroom_output/best_vit.pt \\
      --output_dir ./explain_out \\
      --num_samples 8
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image

from mushroomCNN import MushroomCNN
from mushroomVIT import MushroomVIT


# ─────────────────────────────────────────────────────────────────────────────
# CNN — Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM on a chosen module's output.  We hook the module's forward output
    (activations) and its backward gradient, then form
        cam = ReLU( sum_c [ mean_xy(grad_c) * activation_c ] )
    """
    def __init__(self, model, target_module):
        self.model = model
        self.target = target_module
        self.activations = None
        self.gradients = None
        self._h1 = target_module.register_forward_hook(self._save_act)
        self._h2 = target_module.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        # Ensure the autograd graph reaches our hook by giving the input grad
        x = x.clone().detach().requires_grad_(True)
        logits = self.model(x)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Backprop only the target-class score so gradients reflect that class
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        # weights[c] = mean over spatial dims of grad on channel c
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)            # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1).squeeze(0)            # [H, W]
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, logits.detach().cpu().numpy()[0], target_class

    def remove(self):
        self._h1.remove()
        self._h2.remove()


# ─────────────────────────────────────────────────────────────────────────────
# ViT — attention rollout
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def vit_forward_with_attn(vit, x):
    """
    Manually walk the ViT forward pass while capturing per-layer attention
    weights.  We replicate MushroomVIT.forward but ask MultiheadAttention for
    the weights (need_weights=True, average_attn_weights=False).
    Returns (logits, attentions) where attentions is a list of
        [B, num_heads, N+1, N+1] tensors, one per encoder block.
    """
    B = x.size(0)
    h = vit.patch_embed(x)
    cls = vit.cls_token.expand(B, -1, -1)
    h = torch.cat([cls, h], dim=1)
    h = vit.pos_drop(h + vit.pos_embed)

    attentions = []
    for block in vit.blocks:
        normed = block.norm1(h)
        attn_out, attn_w = block.attn(
            normed, normed, normed,
            need_weights=True,
            average_attn_weights=False,
        )
        attentions.append(attn_w)
        h = h + attn_out
        h = h + block.mlp(block.norm2(h))

    h = vit.norm(h[:, 0])
    logits = vit.head(h)
    return logits, attentions


def attention_rollout(attentions, head_fusion="mean", discard_ratio=0.0):
    """
    Abnar & Zuidema (2020) attention rollout.  Combines attention across all
    layers, accounting for residual connections by adding identity, into a
    single attention map showing where the CLS token effectively looked.

    Returns a 2D numpy heatmap of shape (grid, grid).
    """
    with torch.no_grad():
        N = attentions[0].size(-1)
        device = attentions[0].device
        result = torch.eye(N, device=device)

        for attn in attentions:
            a = attn[0]                                    # [num_heads, N, N]
            if head_fusion == "mean":
                a = a.mean(dim=0)
            elif head_fusion == "max":
                a = a.max(dim=0).values
            else:
                raise ValueError(head_fusion)

            # Optionally drop the lowest-attention edges (often noise)
            if discard_ratio > 0:
                flat = a.view(-1)
                k = int(flat.numel() * discard_ratio)
                if k > 0:
                    thr = flat.kthvalue(k).values
                    a = torch.where(a < thr, torch.zeros_like(a), a)

            # Residual: add identity, renormalise rows
            a = a + torch.eye(N, device=device)
            a = a / a.sum(dim=-1, keepdim=True)

            result = a @ result

        # CLS row, dropping the CLS-to-CLS entry, gives saliency over patches
        mask = result[0, 1:]
        grid = int(round(mask.numel() ** 0.5))
        return mask.reshape(grid, grid).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def overlay_heatmap(img_rgb_01, heatmap, alpha=0.45, size=224):
    """Bilinearly upsample heatmap, normalise, jet-colour, blend with image."""
    h = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    h = F.interpolate(h, size=(size, size), mode="bilinear", align_corners=False)
    h = h.squeeze().numpy()
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)

    cmap = plt.get_cmap("jet")
    h_rgb = cmap(h)[..., :3]
    return np.clip((1 - alpha) * img_rgb_01 + alpha * h_rgb, 0, 1)


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def annotate(ax, model_name, probs, pred_idx, true_idx, class_names):
    pred_name = class_names[pred_idx]
    true_name = class_names[true_idx]
    correct = pred_idx == true_idx
    # Highlight the dangerous failure mode: predicted edible, actually poisonous
    danger = (pred_name.lower() == "edible") and (true_name.lower() == "poisonous")

    if danger:
        colour = "red"
        marker = "✗ DANGER"
    elif correct:
        colour = "green"
        marker = "✓"
    else:
        colour = "orange"
        marker = "✗"

    prob_str = "  ".join(f"{c}: {p:.2f}" for c, p in zip(class_names, probs))
    title = f"{model_name}  {marker}\npred: {pred_name}  ({prob_str})"
    ax.set_title(title, fontsize=10, color=colour)


# ─────────────────────────────────────────────────────────────────────────────
# Sample selection
# ─────────────────────────────────────────────────────────────────────────────

def pick_samples(dataset, indices, cnn, vit, transform, device, num_samples, class_names):
    """
    Run both models on the test split, then choose a balanced set of samples:
    a mix of (correct edible, correct poisonous, incorrect edible, incorrect
    poisonous), prioritising disagreements between models because those are the
    most diagnostic cases for explainability.
    """
    # Cheap forward pass on every test image
    records = []
    cnn.eval(); vit.eval()
    with torch.no_grad():
        for idx in indices:
            path, label = dataset.imgs[idx]
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            cnn_pred = int(cnn(x).argmax(dim=1).item())
            vit_pred = int(vit(x).argmax(dim=1).item())
            records.append(dict(
                idx=idx, path=path, label=label,
                cnn_pred=cnn_pred, vit_pred=vit_pred,
                disagree=(cnn_pred != vit_pred),
            ))

    poison_idx = next((i for i, c in enumerate(class_names)
                       if c.lower() == "poisonous"), 1)

    def take(filt, n):
        return [r for r in records if filt(r)][:n]

    half = max(1, num_samples // 4)
    chosen = []
    chosen += take(lambda r: r["disagree"], half)                                       # disagreements
    chosen += take(lambda r: r["label"] == poison_idx and r["cnn_pred"] != poison_idx,
                   half)                                                                # CNN false negatives (most dangerous)
    chosen += take(lambda r: r["label"] == poison_idx
                              and r["cnn_pred"] == poison_idx
                              and r["vit_pred"] == poison_idx, half)                    # both correct on poisonous
    chosen += take(lambda r: r["label"] != poison_idx
                              and r["cnn_pred"] == poison_idx, half)                    # CNN false positive (over-cautious)

    # de-dup, preserve order, pad with arbitrary remaining records
    seen, uniq = set(), []
    for r in chosen + records:
        if r["idx"] in seen:
            continue
        seen.add(r["idx"])
        uniq.append(r)
        if len(uniq) >= num_samples:
            break
    return uniq[:num_samples]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--cnn_ckpt",   required=True)
    p.add_argument("--vit_ckpt",   required=True)
    p.add_argument("--output_dir", default="./explain_out")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--seed",        type=int, default=42)
    # Model hyperparameters — must match training (defaults match mushroomMain.py)
    p.add_argument("--patch_size",  type=int, default=16)
    p.add_argument("--embed_dim",   type=int, default=256)
    p.add_argument("--num_heads",   type=int, default=8)
    p.add_argument("--depth",       type=int, default=6)
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

    # Eval transform — must match training-time eval pipeline exactly
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Reproduce the same train/val/test split the training script used
    full = datasets.ImageFolder(args.data_dir)
    class_names = full.classes
    n = len(full)
    n_test = int(n * 0.15)
    n_val  = int(n * 0.15)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(args.seed)
    _, _, test_split = torch.utils.data.random_split(
        full, [n_train, n_val, n_test], generator=gen
    )
    test_indices = test_split.indices
    print(f"Test set: {len(test_indices)} images.  classes: {class_names}")

    # Models
    cnn = MushroomCNN(num_classes=len(class_names)).to(device)
    cnn.load_state_dict(torch.load(args.cnn_ckpt, map_location=device))
    cnn.eval()

    vit = MushroomVIT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        num_classes=len(class_names),
    ).to(device)
    vit.load_state_dict(torch.load(args.vit_ckpt, map_location=device))
    vit.eval()

    # Grad-CAM target: output of the full conv feature stack (just before global pool)
    gradcam = GradCAM(cnn, cnn.features)

    # Pick samples
    samples = pick_samples(full, test_indices, cnn, vit,
                            transform, device, args.num_samples, class_names)
    print(f"Selected {len(samples)} samples for explanation.")

    # Build the explanation grid
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 3.4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, rec in enumerate(samples):
        img_pil = Image.open(rec["path"]).convert("RGB")
        img_resized = img_pil.resize((args.img_size, args.img_size))
        img_arr = np.array(img_resized).astype(np.float32) / 255.0

        x = transform(img_pil).unsqueeze(0).to(device)

        # CNN: Grad-CAM on the predicted class (so the heatmap explains *that* call)
        cam, cnn_logits, cnn_pred = gradcam(x)
        cnn_overlay = overlay_heatmap(img_arr, cam, size=args.img_size)
        cnn_probs = softmax(cnn_logits)

        # ViT: forward + rollout
        vit_logits_t, attentions = vit_forward_with_attn(vit, x)
        vit_logits = vit_logits_t.detach().cpu().numpy()[0]
        vit_pred = int(np.argmax(vit_logits))
        roll = attention_rollout(attentions, head_fusion="mean", discard_ratio=0.1)
        vit_overlay = overlay_heatmap(img_arr, roll, size=args.img_size)
        vit_probs = softmax(vit_logits)

        # Plot — col 0: original, col 1: CNN Grad-CAM, col 2: ViT attention
        true_name = class_names[rec["label"]]
        axes[row, 0].imshow(img_arr)
        axes[row, 0].set_title(f"original — true: {true_name}", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(cnn_overlay)
        annotate(axes[row, 1], "CNN Grad-CAM", cnn_probs, cnn_pred,
                  rec["label"], class_names)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(vit_overlay)
        annotate(axes[row, 2], "ViT attention rollout", vit_probs, vit_pred,
                  rec["label"], class_names)
        axes[row, 2].axis("off")

    plt.tight_layout()
    fig_path = out / "explanations.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved explanation grid → {fig_path}")

    gradcam.remove()


if __name__ == "__main__":
    main()
