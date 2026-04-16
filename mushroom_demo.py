"""
mushroom_demo.py
----------------
Interactive Tkinter demo for the mushroom toxicity classifiers.

Usage:
  python mushroom_demo.py \\
      --data_dir ./data \\
      --cnn_ckpt ./mushroom_output/best_cnn.pt \\
      --vit_ckpt ./mushroom_output/best_vit.pt

What it does:
  • shows a real test-set mushroom photo on the left
  • "Next mushroom" picks a fresh random sample from the held-out test split
  • "Run CNN" / "Run ViT" / "Run Both" perform inference and reveal:
        – predicted class (colour-coded for safety)
        – animated softmax probability bars
        – raw logit values
        – ground-truth label
  • dangerous outcomes (predicted EDIBLE when actually POISONOUS) are
    flagged in red so the demo audience immediately sees the failure mode

Tkinter is used to avoid any extra GUI dependencies — only torch, torchvision,
Pillow, numpy, and matplotlib are required (all already needed for training).
"""

import argparse
import random
import threading
from pathlib import Path
from tkinter import Tk, ttk, StringVar, Canvas, Frame, Label, Button
from tkinter import font as tkfont

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mushroomCNN import MushroomCNN
from mushroomVIT import MushroomVIT


IMG_DISPLAY_SIZE = 380   # pixels for the on-screen image
IMG_INPUT_SIZE   = 224   # network input
ANIM_STEPS       = 25    # frames in the bar-fill animation
ANIM_INTERVAL_MS = 18    # ms per frame  →  ~450 ms total


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models(args, num_classes, device):
    cnn = MushroomCNN(num_classes=num_classes).to(device)
    cnn.load_state_dict(torch.load(args.cnn_ckpt, map_location=device))
    cnn.eval()

    vit = MushroomVIT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        num_classes=num_classes,
    ).to(device)
    vit.load_state_dict(torch.load(args.vit_ckpt, map_location=device))
    vit.eval()
    return cnn, vit


def reproduce_test_split(data_dir, img_size, seed):
    """Same seeded split as build_dataloaders so the demo runs on the
    held-out test set the user actually evaluated against."""
    full = datasets.ImageFolder(data_dir)
    n = len(full)
    n_test = int(n * 0.15)
    n_val  = int(n * 0.15)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(seed)
    _, _, test_split = torch.utils.data.random_split(
        full, [n_train, n_val, n_test], generator=gen
    )
    return full, list(test_split.indices)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

class Inferencer:
    def __init__(self, cnn, vit, device, img_size):
        self.cnn = cnn
        self.vit = vit
        self.device = device
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _infer(self, model, pil_img):
        x = self.tf(pil_img).unsqueeze(0).to(self.device)
        logits = model(x).cpu().numpy()[0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return logits, probs

    def cnn_predict(self, pil_img): return self._infer(self.cnn, pil_img)
    def vit_predict(self, pil_img): return self._infer(self.vit, pil_img)


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class MushroomDemoApp:

    BG          = "#fafafa"
    PANEL_BG    = "#ffffff"
    HEADER_BG   = "#1f3550"
    HEADER_FG   = "#ffffff"
    BTN_PRIMARY = "#2c5d8f"
    BTN_NEXT    = "#6c7a89"
    SAFE_GREEN  = "#2e8b57"
    WARN_AMBER  = "#d28e00"
    DANGER_RED  = "#c0392b"

    def __init__(self, root, args):
        self.root = root
        self.args = args

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        # Dataset + models
        self.dataset, self.test_indices = reproduce_test_split(
            args.data_dir, args.img_size, args.seed
        )
        self.class_names = self.dataset.classes
        self.poison_idx = next(
            (i for i, c in enumerate(self.class_names) if c.lower() == "poisonous"),
            len(self.class_names) - 1,
        )

        cnn, vit = load_models(args, len(self.class_names), self.device)
        self.infer = Inferencer(cnn, vit, self.device, args.img_size)

        # State
        self.current_pil = None
        self.current_label = None
        self.current_path = None
        self.unseen = list(self.test_indices)
        random.seed(args.seed)
        random.shuffle(self.unseen)

        self._build_ui()
        self._next_sample()

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("Mushroom Toxicity Classifier — Live Demo")
        self.root.configure(bg=self.BG)
        self.root.geometry("1100x680")

        title_font  = tkfont.Font(family="Helvetica", size=15, weight="bold")
        body_font   = tkfont.Font(family="Helvetica", size=11)
        mono_font   = tkfont.Font(family="Courier",   size=10)

        # ── header
        header = Frame(self.root, bg=self.HEADER_BG, height=46)
        header.pack(fill="x")
        Label(header, text="Mushroom Toxicity Classifier",
              bg=self.HEADER_BG, fg=self.HEADER_FG,
              font=title_font).pack(side="left", padx=18, pady=10)
        Label(header, text=f"device: {self.device}",
              bg=self.HEADER_BG, fg="#aac4dd",
              font=body_font).pack(side="right", padx=18)

        # ── body — split into image (left) and results (right)
        body = Frame(self.root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=14, pady=14)

        # left: image
        left = Frame(body, bg=self.PANEL_BG, bd=1, relief="solid")
        left.pack(side="left", fill="both", expand=False, padx=(0, 14))

        Label(left, text="Test-set sample", bg=self.PANEL_BG,
              font=body_font, fg="#555").pack(pady=(10, 4))

        self.image_canvas = Canvas(
            left, width=IMG_DISPLAY_SIZE, height=IMG_DISPLAY_SIZE,
            bg="#222", highlightthickness=0,
        )
        self.image_canvas.pack(padx=14, pady=4)

        self.truth_var = StringVar(value="")
        Label(left, textvariable=self.truth_var, bg=self.PANEL_BG,
              font=body_font).pack(pady=(8, 4))

        # next-button row
        nav = Frame(left, bg=self.PANEL_BG)
        nav.pack(pady=(6, 14))
        self.next_btn = Button(
            nav, text="↻  Next mushroom",
            command=self._next_sample,
            bg=self.BTN_NEXT, fg="white",
            activebackground="#5a6878", activeforeground="white",
            relief="flat", font=body_font, padx=14, pady=6, cursor="hand2",
        )
        self.next_btn.pack()

        # right: controls + results
        right = Frame(body, bg=self.BG)
        right.pack(side="left", fill="both", expand=True)

        # control buttons
        ctrl = Frame(right, bg=self.BG)
        ctrl.pack(fill="x", pady=(0, 10))
        self.cnn_btn  = self._mk_btn(ctrl, "▶ Run CNN",
                                       lambda: self._run_async("cnn"))
        self.vit_btn  = self._mk_btn(ctrl, "▶ Run ViT",
                                       lambda: self._run_async("vit"))
        self.both_btn = self._mk_btn(ctrl, "▶ Run Both",
                                       lambda: self._run_async("both"))
        for b in (self.cnn_btn, self.vit_btn, self.both_btn):
            b.pack(side="left", padx=(0, 8))

        self.status_var = StringVar(value="ready")
        Label(right, textvariable=self.status_var, bg=self.BG,
              font=body_font, fg="#666").pack(anchor="w", pady=(0, 6))

        # results panel — two stacked sub-panels (CNN / ViT)
        self.cnn_panel = self._make_result_panel(right, "CNN", body_font, mono_font)
        self.cnn_panel["frame"].pack(fill="x", pady=(0, 12))

        self.vit_panel = self._make_result_panel(right, "ViT", body_font, mono_font)
        self.vit_panel["frame"].pack(fill="x")

    def _mk_btn(self, parent, text, cmd):
        return Button(
            parent, text=text, command=cmd,
            bg=self.BTN_PRIMARY, fg="white",
            activebackground="#1e4670", activeforeground="white",
            relief="flat", font=("Helvetica", 11, "bold"),
            padx=14, pady=8, cursor="hand2",
        )

    def _make_result_panel(self, parent, model_name, body_font, mono_font):
        """Build a single model's result panel: header, prediction, softmax
        bar chart (matplotlib), raw logits."""
        frame = Frame(parent, bg=self.PANEL_BG, bd=1, relief="solid")

        header = Frame(frame, bg=self.PANEL_BG)
        header.pack(fill="x", padx=10, pady=(8, 0))
        Label(header, text=model_name, bg=self.PANEL_BG,
              font=("Helvetica", 13, "bold")).pack(side="left")
        pred_var = StringVar(value="—")
        pred_lbl = Label(header, textvariable=pred_var, bg=self.PANEL_BG,
                         font=("Helvetica", 13, "bold"), fg="#555")
        pred_lbl.pack(side="right")

        # matplotlib bar chart for softmax probabilities
        fig = Figure(figsize=(5.2, 1.6), dpi=100, facecolor=self.PANEL_BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.PANEL_BG)
        ax.set_xlim(0, 1.0)
        ax.set_yticks(range(len(self.class_names)))
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel("softmax probability")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        bars = ax.barh(range(len(self.class_names)),
                       [0] * len(self.class_names), color="#bbb")
        text_labels = [
            ax.text(0.005, i, "0.00", va="center", ha="left",
                    fontsize=9, color="#333")
            for i in range(len(self.class_names))
        ]
        fig.tight_layout(pad=0.4)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="x", padx=10, pady=4)

        logits_var = StringVar(value="logits: —")
        Label(frame, textvariable=logits_var, bg=self.PANEL_BG,
              font=mono_font, fg="#444").pack(anchor="w", padx=12, pady=(0, 8))

        return dict(
            frame=frame, pred_var=pred_var, pred_lbl=pred_lbl,
            fig=fig, ax=ax, canvas=canvas, bars=bars,
            text_labels=text_labels, logits_var=logits_var,
        )

    # ── data flow ────────────────────────────────────────────────────────────

    def _next_sample(self):
        if not self.unseen:
            self.unseen = list(self.test_indices)
            random.shuffle(self.unseen)

        idx = self.unseen.pop()
        path, label = self.dataset.imgs[idx]
        self.current_pil = Image.open(path).convert("RGB")
        self.current_label = label
        self.current_path = path

        self._show_image(self.current_pil)
        self.truth_var.set(f"Ground truth:  {self.class_names[label]}")
        self._reset_panel(self.cnn_panel)
        self._reset_panel(self.vit_panel)
        self.status_var.set("ready — choose a model")

    def _show_image(self, pil_img):
        # square-crop preserving aspect ratio for nicer display
        w, h = pil_img.size
        s = min(w, h)
        left = (w - s) // 2
        top  = (h - s) // 2
        cropped = pil_img.crop((left, top, left + s, top + s)) \
                          .resize((IMG_DISPLAY_SIZE, IMG_DISPLAY_SIZE),
                                  Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(cropped)   # keep ref to avoid GC
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

    def _reset_panel(self, panel):
        panel["pred_var"].set("—")
        panel["pred_lbl"].config(fg="#555")
        for bar in panel["bars"]:
            bar.set_width(0)
            bar.set_color("#bbb")
        for t in panel["text_labels"]:
            t.set_text("")
            t.set_x(0.005)
        panel["logits_var"].set("logits: —")
        panel["canvas"].draw_idle()

    # ── inference (threaded so UI stays responsive) ──────────────────────────

    def _run_async(self, which):
        # Disable buttons while running
        for b in (self.cnn_btn, self.vit_btn, self.both_btn, self.next_btn):
            b.config(state="disabled")
        self.status_var.set("predicting…")

        def task():
            try:
                results = {}
                if which in ("cnn", "both"):
                    results["cnn"] = self.infer.cnn_predict(self.current_pil)
                if which in ("vit", "both"):
                    results["vit"] = self.infer.vit_predict(self.current_pil)
                # marshall back to UI thread
                self.root.after(0, lambda: self._show_results(results))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self.status_var.set(f"error: {err}"))
                self.root.after(0, self._reenable_buttons)

        threading.Thread(target=task, daemon=True).start()

    def _reenable_buttons(self):
        for b in (self.cnn_btn, self.vit_btn, self.both_btn, self.next_btn):
            b.config(state="normal")

    def _show_results(self, results):
        for tag, (logits, probs) in results.items():
            panel = self.cnn_panel if tag == "cnn" else self.vit_panel
            self._populate_panel(panel, logits, probs)
        self.status_var.set("done")
        self._reenable_buttons()

    def _populate_panel(self, panel, logits, probs):
        pred_idx = int(np.argmax(probs))
        pred_name = self.class_names[pred_idx]
        true_name = self.class_names[self.current_label]

        # Safety-aware colouring: red when the model says "edible" but the
        # mushroom is actually poisonous.
        is_danger = (pred_name.lower() == "edible"
                     and true_name.lower() == "poisonous")
        is_correct = pred_idx == self.current_label
        if is_danger:
            colour = self.DANGER_RED
            tag = " ⚠ DANGEROUS MISS"
        elif is_correct:
            colour = self.SAFE_GREEN
            tag = " ✓"
        else:
            colour = self.WARN_AMBER
            tag = " ✗ (cautious)"

        panel["pred_var"].set(f"{pred_name}{tag}")
        panel["pred_lbl"].config(fg=colour)

        # Animate bar fill from current width → final probability
        start_widths = [b.get_width() for b in panel["bars"]]
        end_widths   = list(probs)
        bar_colours = [
            self.DANGER_RED if (self.class_names[i].lower() == "edible"
                                 and true_name.lower() == "poisonous"
                                 and i == pred_idx)
            else (self.SAFE_GREEN if i == pred_idx else "#9ab8d8")
            for i in range(len(self.class_names))
        ]
        for bar, c in zip(panel["bars"], bar_colours):
            bar.set_color(c)

        def step(frame_idx):
            t = frame_idx / ANIM_STEPS
            # ease-out curve so it feels snappy
            t_eased = 1 - (1 - t) ** 3
            for i, bar in enumerate(panel["bars"]):
                w = start_widths[i] + (end_widths[i] - start_widths[i]) * t_eased
                bar.set_width(w)
                panel["text_labels"][i].set_text(f"{w:.2f}")
                panel["text_labels"][i].set_x(min(w + 0.01, 0.92))
            panel["canvas"].draw_idle()
            if frame_idx < ANIM_STEPS:
                self.root.after(ANIM_INTERVAL_MS,
                                lambda: step(frame_idx + 1))

        step(1)

        logits_str = "  ".join(f"{c}={v:+.3f}"
                                 for c, v in zip(self.class_names, logits))
        panel["logits_var"].set(f"logits: {logits_str}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--cnn_ckpt",   required=True)
    p.add_argument("--vit_ckpt",   required=True)
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--seed",       type=int, default=42)
    # ViT model hyperparameters (must match training)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim",  type=int, default=256)
    p.add_argument("--num_heads",  type=int, default=8)
    p.add_argument("--depth",      type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    root = Tk()
    MushroomDemoApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
