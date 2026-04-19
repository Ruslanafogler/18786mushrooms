# given trained CNN and ViT and Resnet checkpoints, this script:
# samples a couple test images (correct & incorrect, edible & poisonous)
# computes Grad-CAM for the CNN/Resenet 
# computes attention rollout for the ViT 
# saves a side-by-side panel per image,
# annotates with predicted/true label and softmax probabilities

#   USE CASE ex. goes:
#   python mushroom_explain.py \\
#       --data_dir ./data \\
#       --cnn_ckpt ./mushroom_output/best_cnn.pt \\
#       --vit_ckpt ./mushroom_output/best_vit.pt \\
#       --output_dir ./explain_out \\
#       --num_samples 8

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
from mushroomResNet import MushroomResNet     


class GradCAM:
    #form a RELU of the sum of backpropped gradients from a label and the last level
    #cam = ReLU( sum [ mean(grad) * activation ] 
    #this is supposed to highlight regions of interest that led to the model's choice!!!
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
        x = x.clone().detach().requires_grad_(True)
        logits = self.model(x)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # backprop only the target-class score (GET GRADIETNS FROM THE CLASS CHOICE)
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        #!!! weights are mean over spatial dims of grad on channel c
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)# [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1).squeeze(0)# [H, W]
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, logits.detach().cpu().numpy()[0], target_class

    def remove(self):
        self._h1.remove()
        self._h2.remove()



@torch.no_grad()
def vit_forward_with_attn(vit, x):
    #Manually walk the ViT forward pass while capturing per-layer attention weights.  
    #note that this therefore REPLICATES MushroomVIT.forward in our design :p
    #but ask MultiheadAttention for the weights
    #Returns (logits, attentions) where attentions is a list of
    #[B, num_heads, N+1, N+1] tensors, one per encoder block.
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
    #Abnar & Zuidema (2020) algorithm
    #Combines attention across all layers, accounting for residual connections by adding identity, into a
    #single attention map showing where the CLS token effectively looked.
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

            #Optionally drop the lowest-attention edges
            if discard_ratio > 0:
                flat = a.view(-1)
                k = int(flat.numel() * discard_ratio)
                if k > 0:
                    thr = flat.kthvalue(k).values
                    a = torch.where(a < thr, torch.zeros_like(a), a)

            # RESIDUAL HANDLINGG, we add identity, renormalise rows
            a = a + torch.eye(N, device=device)
            a = a / a.sum(dim=-1, keepdim=True)

            result = a @ result

        #CLS token row, dropping the CLS-to-CLS entry, gives saliency over patches
        mask = result[0, 1:]
        grid = int(round(mask.numel() ** 0.5))
        return mask.reshape(grid, grid).cpu().numpy()



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

def pick_samples(dataset, indices, models, transform, device, num_samples, class_names):
    model_names = list(models.keys())
    for m in models.values():
        m.eval()

    records = []
    with torch.no_grad():
        for idx in indices:
            path, label = dataset.imgs[idx]
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            preds = {name: int(m(x).argmax(dim=1).item())
                     for name, m in models.items()}
            pred_vals = list(preds.values())
            records.append(dict(
                idx=idx, path=path, label=label,
                preds=preds,
                disagree=len(set(pred_vals)) > 1,
            ))

    poison_idx = next((i for i, c in enumerate(class_names)
                       if c.lower() == "poisonous"), 1)

    def take(filt, n):
        return [r for r in records if filt(r)][:n]

    half = max(1, num_samples // 4)
    chosen = []
    #disagreements between any pair of models
    chosen += take(lambda r: r["disagree"], half)
    #any model gives a dangerous false negative
    chosen += take(lambda r: r["label"] == poison_idx and
                   any(p != poison_idx for p in r["preds"].values()), half)
    #all models correct on poisonous
    chosen += take(lambda r: r["label"] == poison_idx and
                   all(p == poison_idx for p in r["preds"].values()), half)
    #any model gives a false positive (over-cautious)
    chosen += take(lambda r: r["label"] != poison_idx and
                   any(p == poison_idx for p in r["preds"].values()), half)

    #de-dup, preserve order, pad with arbitrary remaining records
    seen, uniq = set(), []
    for r in chosen + records:
        if r["idx"] in seen:
            continue
        seen.add(r["idx"])
        uniq.append(r)
        if len(uniq) >= num_samples:
            break
    return uniq[:num_samples]


def compute_heatmap(model, model_name, x, gradcam_map):
    #run one model on one image and return (logits, pred_idx, probs, overlay_data).
    #gradcam_map` is a dict {model_name: GradCAM} for CNN-like models.
    if model_name == "ViT":
        logits_t, attns = vit_forward_with_attn(model, x)
        logits = logits_t.detach().cpu().numpy()[0]
        pred = int(np.argmax(logits))
        probs = softmax(logits)
        heatmap = attention_rollout(attns, head_fusion="mean", discard_ratio=0.1)
        return logits, pred, probs, heatmap
    else:
        #CNN or ResNet — both expose model.features, use Grad-CAM
        gc = gradcam_map[model_name]
        cam, logits, pred = gc(x)
        probs = softmax(logits)
        return logits, pred, probs, cam


def method_label(model_name):
    # Human-readable explainability method name for plot titles
    if model_name == "ViT":
        return "ViT attention rollout"
    return f"{model_name} Grad-CAM"


def parse_args():
    p = argparse.ArgumentParser(
        description="Explainability analysis for mushroom classifiers (CNN / ViT / ResNet)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--output_dir", default="./explain_out")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--seed",        type=int, default=42)

    # Checkpoints — all optional, provide whichever you have
    p.add_argument("--cnn_ckpt",    default=None,
                   help="Path to best_cnn.pt (omit to skip CNN)")
    p.add_argument("--vit_ckpt",    default=None,
                   help="Path to best_vit.pt (omit to skip ViT)")
    p.add_argument("--resnet_ckpt", default=None,
                   help="Path to best_resnet.pt (omit to skip ResNet)")

    # ViT model hyperparameters — must match training
    p.add_argument("--patch_size",  type=int, default=16)
    p.add_argument("--embed_dim",   type=int, default=256)
    p.add_argument("--num_heads",   type=int, default=8)
    p.add_argument("--depth",       type=int, default=6)

    # ResNet model hyperparameters — must match training
    p.add_argument("--resnet_preset", default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--resnet_base_width", type=int, default=64)

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

    #we must match training-time eval pipeline exactly
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(args.data_dir)
    class_names = full.classes
    nc = len(class_names)
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

    models = {}        
    gradcam_map = {}   

    if args.cnn_ckpt:
        cnn = MushroomCNN(num_classes=nc).to(device)
        cnn.load_state_dict(torch.load(args.cnn_ckpt, map_location=device))
        cnn.eval()
        models["CNN"] = cnn
        gradcam_map["CNN"] = GradCAM(cnn, cnn.features)
        print(f"  loaded CNN  ({sum(p.numel() for p in cnn.parameters()):,} params)")

    if args.vit_ckpt:
        vit = MushroomVIT(
            img_size=args.img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            depth=args.depth,
            num_classes=nc,
        ).to(device)
        vit.load_state_dict(torch.load(args.vit_ckpt, map_location=device))
        vit.eval()
        models["ViT"] = vit
        print(f"  loaded ViT  ({sum(p.numel() for p in vit.parameters()):,} params)")

    if args.resnet_ckpt:
        resnet = MushroomResNet(
            preset=args.resnet_preset,
            base_width=args.resnet_base_width,
            num_classes=nc,
        ).to(device)
        resnet.load_state_dict(torch.load(args.resnet_ckpt, map_location=device))
        resnet.eval()
        models["ResNet"] = resnet
        gradcam_map["ResNet"] = GradCAM(resnet, resnet.features)
        print(f"  loaded ResNet ({sum(p.numel() for p in resnet.parameters()):,} params)")

    if not models:
        raise SystemExit(
            "No checkpoints provided.  Pass at least one of "
            "--cnn_ckpt, --vit_ckpt, --resnet_ckpt."
        )

    model_names = list(models.keys())
    print(f"\nModels ready: {model_names}")

    samples = pick_samples(full, test_indices, models,
                            transform, device, args.num_samples, class_names)
    print(f"Selected {len(samples)} samples for explanation.")

    #build grid
    n_rows = len(samples)
    n_cols = 1 + len(model_names)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.6 * n_cols, 3.4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, rec in enumerate(samples):
        img_pil = Image.open(rec["path"]).convert("RGB")
        img_resized = img_pil.resize((args.img_size, args.img_size))
        img_arr = np.array(img_resized).astype(np.float32) / 255.0

        x = transform(img_pil).unsqueeze(0).to(device)

        true_name = class_names[rec["label"]]
        axes[row, 0].imshow(img_arr)
        axes[row, 0].set_title(f"original — true: {true_name}", fontsize=10)
        axes[row, 0].axis("off")

        for col, name in enumerate(model_names, start=1):
            logits, pred, probs, heatmap = compute_heatmap(
                models[name], name, x, gradcam_map)
            overlay = overlay_heatmap(img_arr, heatmap, size=args.img_size)

            axes[row, col].imshow(overlay)
            annotate(axes[row, col], method_label(name), probs, pred,
                      rec["label"], class_names)
            axes[row, col].axis("off")

    plt.tight_layout()
    fig_path = out / "explanations.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved explanation grid → {fig_path}")

    for gc in gradcam_map.values():
        gc.remove()


if __name__ == "__main__":
    main()
