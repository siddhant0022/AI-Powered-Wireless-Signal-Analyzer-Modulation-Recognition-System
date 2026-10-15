"""Train and evaluate the Stage A modulation-recognition baseline."""
import argparse
import csv
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset_loader import MODULATIONS
from model import MultiTaskCNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_training_splits(path):
    """Read only train and validation examples, filtering synthetic anomalies."""
    with np.load(path) as archive:
        required = {"X", "mod", "snr", "split"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        X = archive["X"]
        mod = archive["mod"]
        snr = archive["snr"]
        split = archive["split"]
        normal = archive["anomaly"] == 0 if "anomaly" in archive.files else np.ones(len(mod), bool)

    outputs = {}
    for name, split_id in (("train", 0), ("val", 1)):
        keep = (split == split_id) & normal & (mod >= 0)
        outputs[name] = (
            np.ascontiguousarray(X[keep], dtype=np.float32),
            np.ascontiguousarray(mod[keep], dtype=np.int64),
            np.ascontiguousarray(snr[keep], dtype=np.int16),
        )
    if not len(outputs["train"][0]) or not len(outputs["val"][0]):
        raise ValueError("Training and validation splits must both contain known modulations")
    return outputs


def load_test_split(path):
    """Load test examples only after model training and checkpoint selection finish."""
    with np.load(path) as archive:
        X, mod, snr, split = archive["X"], archive["mod"], archive["snr"], archive["split"]
        normal = archive["anomaly"] == 0 if "anomaly" in archive.files else np.ones(len(mod), bool)
        keep = (split == 2) & normal & (mod >= 0)
        return (
            np.ascontiguousarray(X[keep], dtype=np.float32),
            np.ascontiguousarray(mod[keep], dtype=np.int64),
            np.ascontiguousarray(snr[keep], dtype=np.int16),
        )


def rotate_phase(batch, angles):
    """Rotate each I/Q frame by its own angle without changing its power."""
    cosine, sine = torch.cos(angles)[:, None], torch.sin(angles)[:, None]
    i_samples, q_samples = batch[:, 0], batch[:, 1]
    return torch.stack(
        (i_samples * cosine - q_samples * sine,
         i_samples * sine + q_samples * cosine),
        dim=1,
    )


def _make_loader(data, batch_size, shuffle):
    X, mod, _ = data
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(mod)),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total_count = 0.0, 0, 0
    for X, labels in loader:
        X = X.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            angles = torch.rand(len(X), device=device) * (2 * torch.pi)
            X = rotate_phase(X, angles)
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(X)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        count = len(labels)
        total_loss += loss.item() * count
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += count
    return total_loss / total_count, total_correct / total_count


def _predict(model, X, batch_size, device):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start + batch_size]).to(device)
            predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def _plot_history(history, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Stage A loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=140)
    plt.close(fig)


def _save_test_outputs(model, path, batch_size, device, output_dir):
    # The test set is loaded and evaluated only here, after checkpoint selection.
    X_test, y_test, snr_test = load_test_split(path)
    predictions = _predict(model, X_test, batch_size, device)
    overall_accuracy = float(np.mean(predictions == y_test))
    high_snr = snr_test >= 10
    low_snr = snr_test < 0
    if not high_snr.any() or not low_snr.any():
        raise ValueError("Test set must include both SNR >= 10 dB and SNR < 0 dB")

    accuracy_rows = []
    per_snr = {}
    for snr in sorted(int(value) for value in np.unique(snr_test)):
        mask = snr_test == snr
        accuracy = float(np.mean(predictions[mask] == y_test[mask]))
        per_snr[str(snr)] = {"accuracy": accuracy, "count": int(mask.sum())}
        accuracy_rows.append((snr, accuracy, int(mask.sum())))
    with open(os.path.join(output_dir, "accuracy_vs_snr.csv"), "w", newline="",
              encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("snr_db", "accuracy", "count"))
        writer.writerows(accuracy_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([row[0] for row in accuracy_rows], [row[1] for row in accuracy_rows],
            marker="o", label="Test accuracy")
    ax.axhline(1 / 11, color="gray", linestyle=":", label="Chance (1/11)")
    ax.set(xlabel="SNR (dB)", ylabel="Accuracy", ylim=(0, 1),
           title="Stage A accuracy vs SNR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_vs_snr.png"), dpi=140)
    plt.close(fig)

    high_predictions, high_targets = predictions[high_snr], y_test[high_snr]
    confusion = np.zeros((11, 11), dtype=np.int64)
    np.add.at(confusion, (high_targets, high_predictions), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, label="Frames")
    ax.set(xlabel="Predicted modulation", ylabel="True modulation",
           title="Confusion matrix (test SNR >= 10 dB)")
    ax.set_xticks(range(11), MODULATIONS, rotation=60, ha="right")
    ax.set_yticks(range(11), MODULATIONS)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix_snr_ge_10.png"), dpi=140)
    plt.close(fig)

    summary = {
        "test_accuracy": overall_accuracy,
        "test_accuracy_snr_ge_10_db": float(np.mean(predictions[high_snr] == y_test[high_snr])),
        "test_accuracy_snr_lt_0_db": float(np.mean(predictions[low_snr] == y_test[low_snr])),
        "test_frames": int(len(y_test)),
        "per_snr": per_snr,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join("results", args.tag)
    os.makedirs(output_dir, exist_ok=True)
    splits = load_training_splits(args.data)
    train_loader = _make_loader(splits["train"], args.batch, shuffle=True)
    val_loader = _make_loader(splits["val"], args.batch, shuffle=False)

    model = MultiTaskCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": [],
               "seconds_per_epoch": []}
    best_loss, best_epoch, epochs_without_improvement = float("inf"), 0, 0
    checkpoint_path = os.path.join(output_dir, "best_model.pt")

    print(f"Device: {device}; train={len(splits['train'][0])}; val={len(splits['val'][0])}")
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, _ = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_accuracy = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.perf_counter() - started

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["seconds_per_epoch"].append(elapsed)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f} seconds={elapsed:.1f}"
        )

        if val_loss < best_loss:
            best_loss, best_epoch, epochs_without_improvement = val_loss, epoch, 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch,
                 "val_loss": val_loss},
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
            break

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump({**history, "best_epoch": best_epoch, "best_val_loss": best_loss}, f, indent=2)
    _plot_history(history, output_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    summary = _save_test_outputs(model, args.data, args.batch, device, output_dir)
    print(
        f"Test accuracy={summary['test_accuracy']:.4f}; "
        f"SNR >= 10 dB={summary['test_accuracy_snr_ge_10_db']:.4f}; "
        f"SNR < 0 dB={summary['test_accuracy_snr_lt_0_db']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join("data", "radioml_2016a.npz"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tag", default="stageA")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.patience < 1:
        parser.error("--epochs, --batch, and --patience must be positive")
    train(args)


if __name__ == "__main__":
    main()
