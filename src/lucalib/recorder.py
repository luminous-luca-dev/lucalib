"""Helpers for collecting per-epoch training metrics."""

from __future__ import annotations

from typing import Any


class History:
    """Collect train/validation loss and accuracy for each epoch.

    Call :meth:`append` once per epoch. Values can be Python numbers or scalar
    tensors (for example, values returned by PyTorch's ``loss.item()``).
    """

    def __init__(self) -> None:
        self.data: dict[str, list[float]] = {
            "train_loss": [],
            "valid_loss": [],
            "train_acc": [],
            "valid_acc": [],
        }

    @staticmethod
    def _number(value: Any) -> float:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def append(
        self,
        *,
        train_loss: Any,
        valid_loss: Any,
        train_acc: Any,
        valid_acc: Any,
    ) -> None:
        """Record the four mean metrics for one completed epoch."""
        self.data["train_loss"].append(self._number(train_loss))
        self.data["valid_loss"].append(self._number(valid_loss))
        self.data["train_acc"].append(self._number(train_acc))
        self.data["valid_acc"].append(self._number(valid_acc))

    def __getitem__(self, key: str) -> list[float]:
        return self.data[key]

    def as_dict(self) -> dict[str, list[float]]:
        """Return a plain dictionary suitable for :func:`plot_history`."""
        return {key: values.copy() for key, values in self.data.items()}
