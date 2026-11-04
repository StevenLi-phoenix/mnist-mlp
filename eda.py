"""Exploratory data analysis of MNIST (reads the raw idx files, numpy + matplotlib only)."""
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eda")

RAW, OUT = Path("data/MNIST/raw"), Path("report")
INK, ACCENT, MUTED = "#222222", "#c0392b", "#8a8a8a"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "savefig.bbox": "tight"})


def idx(name: str) -> np.ndarray:
    raw = np.fromfile(RAW / name, dtype=np.uint8)
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i: 8 + 4 * i].tobytes(), "big") for i in range(ndim))
    return raw[4 + 4 * ndim:].reshape(shape)


def samples(x: np.ndarray, y: np.ndarray, per_class: int = 12) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(10, per_class + 1, figsize=(6.6, 5.6),
                             gridspec_kw={"width_ratios": [0.6] + [1] * per_class})
    for d in range(10):
        axes[d, 0].axis("off")
        axes[d, 0].text(0.7, 0.5, str(d), ha="right", va="center", fontsize=12, color=INK, weight="bold")
        for c, i in enumerate(rng.choice(np.where(y == d)[0], per_class, replace=False)):
            axes[d, c + 1].imshow(x[i], cmap="gray_r")
            axes[d, c + 1].axis("off")
    fig.tight_layout(pad=0.2, h_pad=0.15, w_pad=0.1)
    fig.savefig(OUT / "eda_samples.pdf")
    plt.close(fig)


def stats(xtr, ytr, yva, xte, yte) -> dict:
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.8, 2.4), gridspec_kw={"width_ratios": [1.2, 1, 1.2]})
    ctr, cva, cte = (np.bincount(v, minlength=10) for v in (ytr, yva, yte))
    for off, cnt, col, lab in ((-0.27, ctr, INK, "train"), (0, cva, MUTED, "val"), (0.27, cte, ACCENT, "test")):
        a.bar(np.arange(10) + off, 100 * cnt / cnt.sum(), 0.27, color=col, label=lab)
    a.set(xticks=range(10), xlabel="digit", ylabel="share of split (%)", title="Class balance", ylim=(0, 14.5))
    a.legend(frameon=False, fontsize=7, ncol=3, loc="upper center")

    flat = xtr.reshape(-1)
    hist = np.bincount(flat, minlength=256)
    b.bar(np.arange(256), hist / flat.size, width=1.0, color=INK)
    b.set(yscale="log", xlabel="pixel value", ylabel="fraction of pixels", title="Pixel intensities")

    ink = xtr.reshape(len(xtr), -1).astype(np.float32).sum(1) / 255  # total ink per image
    data = [ink[ytr == d] for d in range(10)]
    c.boxplot(data, widths=0.6, showfliers=False, medianprops=dict(color=ACCENT),
              boxprops=dict(color=INK), whiskerprops=dict(color=INK), capprops=dict(color=INK))
    c.set(xticklabels=range(10), xlabel="digit", ylabel="ink (sum of pixels / 255)", title="Ink per image")
    fig.tight_layout()
    fig.savefig(OUT / "eda_stats.pdf")
    plt.close(fig)

    return {"n_train": len(ytr), "n_val": len(yva), "n_test": len(yte), "train_counts": ctr.tolist(),
            "val_counts": cva.tolist(), "test_counts": cte.tolist(),
            "zero_frac": float((flat == 0).mean()), "saturated_frac": float((flat >= 250).mean()),
            "mean": float(flat.mean() / 255), "std": float(flat.std() / 255),
            "ink_median": [float(np.median(v)) for v in data],
            "bbox_hits_border_frac": float((xtr[:, [0, -1], :].any(2).any(1) | xtr[:, :, [0, -1]].any(1).any(1)).mean())}


def mean_images(x: np.ndarray, y: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 10, figsize=(6.6, 0.85))
    for d, ax in enumerate(axes):
        ax.imshow(x[y == d].mean(0), cmap="gray_r")
        ax.set_title(str(d), fontsize=9, pad=2)
        ax.axis("off")
    fig.tight_layout(pad=0.1, w_pad=0.1)
    fig.savefig(OUT / "eda_mean.pdf")
    plt.close(fig)


def main() -> None:
    xall, yall = idx("train-images-idx3-ubyte"), idx("train-labels-idx1-ubyte")
    split = np.load("split.npz")  # written by train.py: the 55k/5k train/val indices
    xtr, ytr = xall[split["train_idx"]], yall[split["train_idx"]]
    yva = yall[split["val_idx"]]
    xte, yte = idx("t10k-images-idx3-ubyte"), idx("t10k-labels-idx1-ubyte")
    log.info("train %s val %s test %s dtype %s", xtr.shape, yva.shape, xte.shape, xtr.dtype)
    samples(xtr, ytr)
    info = stats(xtr, ytr, yva, xte, yte)
    mean_images(xtr, ytr)
    json.dump(info, open(OUT / "eda.json", "w"), indent=2)
    log.info("%s", json.dumps({k: v for k, v in info.items() if k != "ink_median"}))
    log.info("ink median per digit: %s", [round(v, 1) for v in info["ink_median"]])


if __name__ == "__main__":
    main()
