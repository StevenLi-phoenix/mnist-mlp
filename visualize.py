"""Make report figures + metrics.json from history.json and predictions.npz (numpy + matplotlib only)."""
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("viz")

OUT = Path("report")
OUT.mkdir(exist_ok=True)
INK, ACCENT, MUTED = "#222222", "#c0392b", "#8a8a8a"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "savefig.bbox": "tight"})


def curves(hist: list[dict], best_epoch: int) -> None:
    ep = [h["epoch"] for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    a.plot(ep, [h["train_loss"] for h in hist], color=INK, marker="o", ms=3, label="train")
    a.plot(ep, [h["val_loss"] for h in hist], color=ACCENT, marker="s", ms=3, label="validation")
    a.axvline(best_epoch, color=MUTED, ls="--", lw=0.8)
    a.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    a.legend(frameon=False)
    va = [100 * h["val_acc"] for h in hist]
    b.plot(ep, va, color=ACCENT, marker="s", ms=3)
    b.axvline(best_epoch, color=MUTED, ls="--", lw=0.8)
    b.annotate(f"selected: epoch {best_epoch}\n{va[best_epoch - 1]:.2f}%", (best_epoch, va[best_epoch - 1]),
               xytext=(-62, -42), textcoords="offset points", color=INK, fontsize=8,
               arrowprops=dict(arrowstyle="-", color=MUTED))
    b.set(xlabel="epoch", ylabel="validation accuracy (%)", title="Validation accuracy")
    fig.tight_layout()
    fig.savefig(OUT / "curves.pdf")
    plt.close(fig)


def confusion(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    off = cm.copy()
    np.fill_diagonal(off, 0)  # colour only the errors; diagonal stays white so mistakes stand out
    ax.imshow(off, cmap="Reds", vmin=0, vmax=max(off.max(), 1))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color=MUTED if i == j else (INK if off[i, j] < off.max() * 0.6 else "white"))
    ax.set(xticks=range(10), yticks=range(10), xlabel="predicted", ylabel="true")
    ax.spines[:].set_visible(False)
    fig.savefig(OUT / "confusion.pdf")
    plt.close(fig)


def misclassified(images, labels, preds, conf) -> None:
    wrong = np.where(preds != labels)[0]
    pick = wrong[np.argsort(-conf[wrong])][:24]  # most confident mistakes first
    fig, axes = plt.subplots(3, 8, figsize=(6.6, 2.9))
    for ax, i in zip(axes.ravel(), pick):
        ax.imshow(images[i], cmap="gray_r")
        ax.set_title(f"{labels[i]}$\\rightarrow${preds[i]} ({conf[i]:.2f})", fontsize=7, color=ACCENT)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "misclassified.pdf")
    plt.close(fig)


def confusion_examples(cm, images, labels, preds, conf, n_pairs: int = 5, n_show: int = 5) -> None:
    """One row per most-common confusion: label on the left, the most confident examples on the right."""
    off = sorted(((cm[i, j], i, j) for i in range(10) for j in range(10) if i != j), reverse=True)[:n_pairs]
    fig, axes = plt.subplots(n_pairs, n_show + 1, figsize=(6.6, 1.15 * n_pairs),
                             gridspec_kw={"width_ratios": [1.6] + [1] * n_show})
    for r, (count, t, p) in enumerate(off):
        idx = np.where((labels == t) & (preds == p))[0]
        idx = idx[np.argsort(-conf[idx])][:n_show]
        lab = axes[r, 0]
        lab.axis("off")
        lab.text(0.95, 0.5, f"true {t}\npredicted {p}\n{count} images", ha="right", va="center",
                 fontsize=9, color=INK, linespacing=1.5)
        for c in range(n_show):
            ax = axes[r, c + 1]
            ax.axis("off")
            if c < len(idx):
                ax.imshow(images[idx[c]], cmap="gray_r")
                ax.set_title(f"{conf[idx[c]]:.2f}", fontsize=7, color=ACCENT, pad=2)
    fig.tight_layout(h_pad=0.4, w_pad=0.2)
    fig.savefig(OUT / "confusion_examples.pdf")
    plt.close(fig)


def confidence(conf, correct) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    bins = np.linspace(0.3, 1.0, 36)
    ax.hist(conf[correct], bins=bins, color=INK, alpha=0.8, label=f"correct (n={correct.sum()})")
    ax.hist(conf[~correct], bins=bins, color=ACCENT, alpha=0.9, label=f"wrong (n={(~correct).sum()})")
    ax.set_yscale("log")
    ax.set(xlabel="max softmax probability", ylabel="# test images (log)")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "confidence.pdf")
    plt.close(fig)


def filters(w1: np.ndarray) -> None:
    fig, axes = plt.subplots(4, 8, figsize=(6.0, 3.2))
    for ax, w in zip(axes.ravel(), w1[:32]):
        m = np.abs(w).max()
        ax.imshow(w.reshape(28, 28), cmap="RdBu_r", vmin=-m, vmax=m)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "filters.pdf")
    plt.close(fig)


def main() -> None:
    h = json.load(open("history.json"))
    d = np.load("predictions.npz")
    probs, labels, images, w1 = d["probs"], d["labels"], d["images"], d["w1"]
    preds, conf = probs.argmax(1), probs.max(1)
    correct = preds == labels
    cm = np.zeros((10, 10), int)
    np.add.at(cm, (labels, preds), 1)

    per_class = []
    for c in range(10):
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c].sum() - cm[c, c]
        p, r = tp / (tp + fp), tp / (tp + fn)
        per_class.append({"digit": c, "support": int(cm[c].sum()), "precision": p, "recall": r,
                          "f1": 2 * p * r / (p + r)})
    off = [(int(cm[i, j]), i, j) for i in range(10) for j in range(10) if i != j]
    metrics = {"acc": float(correct.mean()), "errors": int((~correct).sum()), "per_class": per_class,
               "top_confusions": sorted(off, reverse=True)[:5],
               "mean_conf_correct": float(conf[correct].mean()), "mean_conf_wrong": float(conf[~correct].mean()),
               "wrong_conf_gt_90": int(((~correct) & (conf > 0.9)).sum())}
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)
    log.info("acc=%.4f errors=%d top_confusions=%s", metrics["acc"], metrics["errors"], metrics["top_confusions"])

    curves(h["history"], h["best_epoch"])
    confusion(cm)
    misclassified(images, labels, preds, conf)
    confusion_examples(cm, images, labels, preds, conf)
    confidence(conf, correct)
    filters(w1)
    log.info("figures written to %s: %s", OUT, sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
