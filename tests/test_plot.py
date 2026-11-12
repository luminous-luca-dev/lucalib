"""Tests for lucalib.plot.plot_history.

matplotlib is set to a non-interactive backend so these tests run headlessly.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # must be set before any other matplotlib import

import matplotlib.pyplot as plt
import pytest
from pathlib import Path

from lucalib import History, plot_history


def _history_dict(n: int = 5) -> dict:
    h = History()
    for i in range(n):
        h.append(
            train_loss=1.0 - i * 0.1,
            valid_loss=1.05 - i * 0.1,
            train_acc=0.5 + i * 0.08,
            valid_acc=0.48 + i * 0.08,
        )
    return h.as_dict()


class TestPlotHistoryBasic:
    def test_returns_fig_and_path(self, tmp_path):
        fig, saved = plot_history(_history_dict(), output_dir=tmp_path, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(saved, Path)
        assert saved.exists()

    def test_png_extension(self, tmp_path):
        _, saved = plot_history(_history_dict(), output_dir=tmp_path, show=False)
        assert saved.suffix == ".png"

    def test_custom_filename(self, tmp_path):
        _, saved = plot_history(
            _history_dict(),
            output_dir=tmp_path,
            filename="myplot.png",
            show=False,
        )
        assert saved.name == "myplot.png"

    def test_output_dir_created(self, tmp_path):
        subdir = tmp_path / "nested" / "dir"
        _, saved = plot_history(_history_dict(), output_dir=subdir, show=False)
        assert saved.parent.exists()

    def test_all_metadata_args(self, tmp_path):
        """Should not raise when all optional keyword arguments are passed."""
        fig, _ = plot_history(
            _history_dict(),
            lr=0.001,
            n_epochs=5,
            batch_size=32,
            hidden_dims=128,
            best_val_acc=0.92,
            comment="integration test",
            output_dir=tmp_path,
            show=False,
        )
        assert fig is not None

    def test_missing_key_raises(self, tmp_path):
        bad = {"train_loss": [1.0], "valid_loss": [1.0], "train_acc": [0.5]}
        with pytest.raises(KeyError):
            plot_history(bad, output_dir=tmp_path, show=False)

    def test_unequal_length_raises(self, tmp_path):
        bad = {
            "train_loss": [1.0, 0.9],
            "valid_loss": [1.0],
            "train_acc": [0.5, 0.6],
            "valid_acc": [0.5],
        }
        with pytest.raises(ValueError):
            plot_history(bad, output_dir=tmp_path, show=False)

    def test_saved_path_reusable(self, tmp_path):
        """Returned path must be a valid, openable file."""
        _, saved = plot_history(_history_dict(), output_dir=tmp_path, show=False)
        with open(saved, "rb") as fh:
            header = fh.read(4)
        # PNG magic bytes: 0x89 P N G
        assert header == b"\x89PNG"

