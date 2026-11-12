"""Regression tests for lucalib.recorder.History."""

import pytest

from lucalib import History


def _make_history(n: int = 3) -> History:
    h = History()
    for i in range(n):
        h.append(
            train_loss=1.0 - i * 0.1,
            valid_loss=1.1 - i * 0.1,
            train_acc=0.5 + i * 0.1,
            valid_acc=0.45 + i * 0.1,
        )
    return h


class TestHistoryAppend:
    def test_length_increases(self):
        h = _make_history(4)
        assert len(h["train_loss"]) == 4

    def test_values_stored_as_float(self):
        h = _make_history(1)
        assert isinstance(h["train_loss"][0], float)

    def test_tensor_like_converted(self):
        """Objects with `.item()` (e.g. PyTorch scalars) must be accepted."""

        class FakeScalar:
            def item(self):
                return 0.42

        h = History()
        h.append(
            train_loss=FakeScalar(),
            valid_loss=FakeScalar(),
            train_acc=FakeScalar(),
            valid_acc=FakeScalar(),
        )
        assert h["train_loss"][0] == pytest.approx(0.42)

    def test_as_dict_returns_copy(self):
        h = _make_history(2)
        d = h.as_dict()
        d["train_loss"].append(999.0)
        # original must be unaffected
        assert 999.0 not in h["train_loss"]

    def test_getitem_known_keys(self):
        h = _make_history(2)
        for key in ("train_loss", "valid_loss", "train_acc", "valid_acc"):
            assert len(h[key]) == 2


class TestHistoryEdge:
    def test_empty_history(self):
        h = History()
        assert h["train_loss"] == []

    def test_as_dict_empty(self):
        h = History()
        d = h.as_dict()
        assert all(v == [] for v in d.values())

