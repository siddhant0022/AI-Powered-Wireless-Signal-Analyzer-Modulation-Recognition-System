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
from model import (
    MultiTaskCNN,
    checkpoint_payload,
    load_model_checkpoint,
    scale_snr_target,
)


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


def _make_loader(data, batch_size, shuffle, device="cpu", use_snr=False):
    X, mod, snr = data
    tensors = [
        torch.as_tensor(X, dtype=torch.float32, device=device),
        torch.as_tensor(mod, dtype=torch.long, device=device),
    ]
    if use_snr:
        snr_db = torch.as_tensor(snr, dtype=torch.float32, device=device)
        tensors.append(scale_snr_target(snr_db))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _run_epoch(model, loader, criterion, device, optimizer=None, use_snr=False,
               lambda_snr=1.0):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total_snr_error, total_count = 0.0, 0, 0.0, 0
    for batch in loader:
        X, labels = batch[:2]
        X = X.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_snr:
            snr_target = batch[2].to(device, non_blocking=True)
        if training:
            angles = torch.rand(len(X), device=device) * (2 * torch.pi)
            X = rotate_phase(X, angles)
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model(X)
            if use_snr:
                logits, snr_pred = output
                snr_pred = snr_pred.squeeze(1)
                loss = criterion(logits, labels) + lambda_snr * nn.functional.smooth_l1_loss(
                    snr_pred, snr_target
                )
                total_snr_error += (
                    (snr_pred.detach() - snr_target).abs().sum().item() * 20
                )
            else:
                logits = output
                loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        count = len(labels)
        total_loss += loss.item() * count
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += count
    metrics = (total_loss / total_count, total_correct / total_count)
    if use_snr:
        return (*metrics, total_snr_error / total_count)
    return metrics


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


def _capture_rng_states():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0], numpy_state[1].tolist(), numpy_state[2],
            numpy_state[3], numpy_state[4],
        ],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_states(states):
    random.setstate(states["python"])
    numpy_state = states["numpy"]
    np.random.set_state((
        numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
        int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4]),
    ))
    torch.set_rng_state(states["torch_cpu"].cpu())
    if torch.cuda.is_available() and states["torch_cuda"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in states["torch_cuda"]])


def _write_history_csv(path, rows):
    fields = (
        "epoch", "train_loss", "val_loss", "val_mod_acc",
        "val_snr_mae_db", "epoch_seconds",
    )
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_history_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as stream:
        return [
            {
                **row,
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "val_mod_acc": float(row["val_mod_acc"]),
                "val_snr_mae_db": (
                    float(row["val_snr_mae_db"]) if row["val_snr_mae_db"] else ""
                ),
                "epoch_seconds": float(row["epoch_seconds"]),
            }
            for row in csv.DictReader(stream)
        ]


def _save_checkpoint(path, payload):
    temporary_path = f"{path}.tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def train(args):
    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset_paths = {
        "simulated": os.path.join("data", "signal_dataset.npz"),
        "radioml": os.path.join("data", "radioml_2016a.npz"),
    }
    data_path = args.data or dataset_paths[args.dataset]
    output_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    last_path = os.path.join(output_dir, "last.pt")
    best_path = os.path.join(output_dir, "best.pt")
    best_model_path = os.path.join(output_dir, "best_model.pt")
    history_path = os.path.join(output_dir, "history.csv")
    config_path = os.path.join(output_dir, "config.json")

    patience = args.patience
    if patience is None and not args.use_snr:
        patience = 8
    training_config = {
        "dataset": args.dataset,
        "data_path": os.path.abspath(data_path),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.wd,
        "lambda_snr": args.lambda_snr,
        "use_snr": args.use_snr,
        "seed": args.seed,
        "device": str(device),
    }

    splits = load_training_splits(data_path)
    train_loader = _make_loader(
        splits["train"], args.batch_size, shuffle=True, device=device,
        use_snr=args.use_snr,
    )
    val_loader = _make_loader(
        splits["val"], args.batch_size, shuffle=False, device=device,
        use_snr=args.use_snr,
    )

    model = MultiTaskCNN(use_snr=args.use_snr).to(device)
    model_config = model.get_config()
    run_config = {
        **vars(args),
        "data_path": os.path.abspath(data_path),
        "device": str(device),
        "effective_patience": patience,
        "model": model_config,
        "checkpoint_selection": (
            "val_mod_acc - 0.5 * val_snr_mae_db / 20"
            if args.use_snr else "lowest validation modulation cross-entropy"
        ),
    }
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )
    history_rows = []
    start_epoch = 1
    best_score, best_epoch, epochs_without_improvement = float("-inf"), 0, 0

    if args.resume and os.path.isfile(last_path):
        loaded_model, checkpoint = load_model_checkpoint(last_path, map_location="cpu")
        if checkpoint.get("training_config") != training_config:
            raise ValueError("Resume configuration does not match the saved run")
        if not all(key in checkpoint for key in (
            "optimizer_state_dict", "scheduler_state_dict", "rng_states", "epoch",
        )):
            raise ValueError(f"{last_path} is missing resumable training state")
        model.load_state_dict(loaded_model.state_dict())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history_rows = checkpoint.get("history_rows", _read_history_csv(history_path))
        _restore_rng_states(checkpoint["rng_states"])

    with open(config_path, "w", encoding="utf-8") as stream:
        json.dump(run_config, stream, indent=2)
    _write_history_csv(history_path, history_rows)

    print(
        f"Device: {device}; train={len(splits['train'][0])}; "
        f"val={len(splits['val'][0])}; use_snr={args.use_snr}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer,
            use_snr=args.use_snr, lambda_snr=args.lambda_snr,
        )
        val_metrics = _run_epoch(
            model, val_loader, criterion, device, use_snr=args.use_snr,
            lambda_snr=args.lambda_snr,
        )
        train_loss = train_metrics[0]
        val_loss, val_mod_acc = val_metrics[:2]
        val_snr_mae_db = val_metrics[2] if args.use_snr else ""
        scheduler.step(val_loss)
        elapsed = time.perf_counter() - started

        score = (
            val_mod_acc - 0.5 * val_snr_mae_db / 20
            if args.use_snr else -val_loss
        )
        improved = score > best_score
        if improved:
            best_score, best_epoch, epochs_without_improvement = score, epoch, 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mod_acc": val_mod_acc,
            "val_snr_mae_db": val_snr_mae_db,
            "epoch_seconds": elapsed,
        }
        history_rows.append(row)
        _write_history_csv(history_path, history_rows)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mod_acc={val_mod_acc:.4f} "
            f"val_snr_mae_db={val_snr_mae_db if args.use_snr else 'n/a'} "
            f"seconds={elapsed:.1f}"
        )

        payload = checkpoint_payload(
            model,
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            epoch=epoch,
            best_score=best_score,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            rng_states=_capture_rng_states(),
            training_config=training_config,
            history_rows=history_rows,
            val_loss=val_loss,
            val_mod_acc=val_mod_acc,
            val_snr_mae_db=val_snr_mae_db,
        )
        if improved:
            _save_checkpoint(best_path, payload)
            if not args.use_snr:
                _save_checkpoint(best_model_path, payload)
        _save_checkpoint(last_path, payload)

        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
            break

    history = {
        "epoch": [row["epoch"] for row in history_rows],
        "train_loss": [row["train_loss"] for row in history_rows],
        "val_loss": [row["val_loss"] for row in history_rows],
        "val_accuracy": [row["val_mod_acc"] for row in history_rows],
        "seconds_per_epoch": [row["epoch_seconds"] for row in history_rows],
    }
    if args.use_snr:
        history["val_snr_mae_db"] = [row["val_snr_mae_db"] for row in history_rows]
    history_summary = {
        **history,
        "best_epoch": best_epoch,
        "best_score": best_score,
    }
    if not args.use_snr:
        history_summary["best_val_loss"] = -best_score
    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as stream:
        json.dump(history_summary, stream, indent=2)
    _plot_history(history, output_dir)

    if not args.use_snr:
        model, _ = load_model_checkpoint(best_model_path, map_location=device)
        summary = _save_test_outputs(
            model, data_path, args.batch_size, device, output_dir
        )
        print(
            f"Test accuracy={summary['test_accuracy']:.4f}; "
            f"SNR >= 10 dB={summary['test_accuracy_snr_ge_10_db']:.4f}; "
            f"SNR < 0 dB={summary['test_accuracy_snr_lt_0_db']:.4f}"
        )
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("simulated", "radioml"), default="radioml")
    parser.add_argument("--data", help="optional explicit dataset NPZ path")
    parser.add_argument("--use-snr", action="store_true")
    parser.add_argument("--lambda-snr", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", "--batch", dest="batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--device")
    parser.add_argument("--run-name", "--tag", dest="run_name", default="stageA")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.patience is not None and args.patience < 1:
        parser.error("--patience must be positive when provided")
    if args.lr <= 0 or args.wd < 0 or not np.isfinite(args.lambda_snr) or args.lambda_snr < 0:
        parser.error("--lr must be positive; --wd and --lambda-snr must be non-negative")
    return train(args)


if __name__ == "__main__":
    main()
