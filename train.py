"""MNIST MLP with 3 hidden layers: 784 -> 512 -> 256 -> 128 -> 10.

Train/validation/test protocol: 60k training images are split 55k/5k (train/val, seed 0). The epoch is
selected on validation accuracy; the test set is evaluated exactly once, on the selected checkpoint.
"""
import copy
import json
import logging
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mnist")

EPOCHS, BATCH, LR, N_VAL, SEED = 10, 128, 1e-3, 5_000, 0


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += nn.functional.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()
    return loss_sum / n, correct / n


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_train = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_ds, val_ds = random_split(full_train, [len(full_train) - N_VAL, N_VAL],
                                    generator=torch.Generator().manual_seed(SEED))
    np.savez("split.npz", train_idx=np.array(train_ds.indices), val_idx=np.array(val_ds.indices))
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1000)
    test_dl = DataLoader(test_ds, batch_size=1000)

    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("device=%s params=%d train=%d val=%d test=%d",
             device, n_params, len(train_ds), len(val_ds), len(test_ds))

    history: list[dict] = []
    best = {"acc": -1.0, "loss": float("inf"), "epoch": 0, "state": None}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
        val_loss, val_acc = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": run_loss / len(train_ds),
                        "val_loss": val_loss, "val_acc": val_acc, "seconds": time.time() - t0})
        improved = val_acc > best["acc"] or (val_acc == best["acc"] and val_loss < best["loss"])
        if improved:
            best = {"acc": val_acc, "loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        log.info("epoch %2d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f%s (%.1fs)", epoch, EPOCHS,
                 run_loss / len(train_ds), val_loss, val_acc, " *" if improved else "", time.time() - t0)

    model.load_state_dict(best["state"])
    test_loss, test_acc = evaluate(model, test_dl, device)  # the only time the test set is used
    log.info("selected epoch %d (val_acc=%.4f) -> test_loss=%.4f test_acc=%.4f",
             best["epoch"], best["acc"], test_loss, test_acc)

    with open("history.json", "w") as f:
        json.dump({"params": n_params, "device": str(device), "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                   "seed": SEED, "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
                   "best_epoch": best["epoch"], "val_acc": best["acc"], "val_loss": best["loss"],
                   "test_acc": test_acc, "test_loss": test_loss, "history": history}, f, indent=2)
    with torch.no_grad():
        probs = torch.cat([model(x.to(device)).softmax(1).cpu() for x, _ in test_dl]).numpy()
    np.savez_compressed("predictions.npz", probs=probs, labels=test_ds.targets.numpy(),
                        images=test_ds.data.numpy(), w1=model.net[1].weight.detach().cpu().numpy())
    torch.save(model.state_dict(), "mnist_mlp.pt")
    log.info("saved history.json, predictions.npz, split.npz, mnist_mlp.pt")


if __name__ == "__main__":
    main()
