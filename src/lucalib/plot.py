"""Training-history plotting utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt


_REQUIRED_KEYS = ("train_loss", "valid_loss", "train_acc", "valid_acc")


def plot_history(
    history: Mapping[str, Any],
    *,
    lr: float | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    hidden_dims: Any | None = None,
    best_val_acc: float | None = None,
    comment: str | None = None,
    output_dir: str | Path | None = None,
    filename: str | None = None,
    show: bool = True,
    dpi: int = 150,
):
    """Plot loss and accuracy history, optionally save it, and return the figure.

    ``history`` must contain ``train_loss``, ``valid_loss``, ``train_acc`` and
    ``valid_acc`` sequences of equal length. When ``output_dir`` is supplied,
    the plot is saved there as a timestamped PNG unless ``filename`` is given.
    """
    missing = [key for key in _REQUIRED_KEYS if key not in history]
    if missing:
        raise KeyError(f"history is missing required key(s): {', '.join(missing)}")

    lengths = {key: len(history[key]) for key in _REQUIRED_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"history series must have equal lengths; got {lengths}")

    epochs = range(lengths["train_loss"])
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    ax1.plot(epochs, history["train_loss"], label="Train Loss", linestyle="-", color="green")
    ax1.plot(epochs, history["valid_loss"], label="Valid Loss", linestyle="--", color="gold")
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", linestyle="-", color="red")
    ax2.plot(epochs, history["valid_acc"], label="Valid Accuracy", linestyle="--", color="blue")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.grid(True, alpha=0.3)

    metadata = []
    if lr is not None:
        metadata.append(f"LR: {lr}")
    if n_epochs is not None:
        metadata.append(f"N_EPOCH: {n_epochs}")
    if batch_size is not None:
        metadata.append(f"BatchSize: {batch_size}")
    if hidden_dims is not None:
        metadata.append(f"HiddenDims: {hidden_dims}")
    if best_val_acc is not None:
        metadata.append(f"Best validation accuracy: {best_val_acc:.4f}")
    if comment:
        metadata.append(comment)
    if metadata:
        fig.suptitle("\n".join(metadata), fontsize=12, y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 1.00] if metadata else None)

    directory = Path(output_dir) if output_dir is not None else Path.cwd().parent / "record"
    directory.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    save_path = directory / filename
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    result = (fig, save_path)

    if show:
        plt.show()
    return result
