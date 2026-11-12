"""Tests for lucalib.discord_notify.send_discord.

HTTP requests are mocked via unittest.mock so no real Webhook is called.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lucalib import DiscordError, send_discord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_response(status: int = 204) -> MagicMock:
    """Return a context-manager mock that mimics urllib's HTTP response."""
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _http_error(code: int, body: str = "error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://discord.test",
        code=code,
        msg="Error",
        hdrs=None,  # type: ignore[arg-type]
        fp=BytesIO(body.encode()),
    )


FAKE_URL = "https://discord.com/api/webhooks/fake/url"


# ---------------------------------------------------------------------------
# Text-only tests
# ---------------------------------------------------------------------------

class TestSendDiscordTextOnly:
    def test_text_only_success(self):
        with patch("urllib.request.urlopen", return_value=_fake_response(204)) as mock_open:
            send_discord("hello", webhook_url=FAKE_URL)
        mock_open.assert_called_once()
        req = mock_open.call_args[0][0]
        body = json.loads(req.data.decode())
        assert body["content"] == "hello"

    def test_no_url_raises_value_error(self):
        env = {k: v for k, v in os.environ.items() if k != "DISCORD_WEBHOOK_URL"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Webhook URL"):
                send_discord("hi")

    def test_env_var_url_used(self):
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": FAKE_URL}):
            with patch("urllib.request.urlopen", return_value=_fake_response(204)):
                send_discord("from env")  # must not raise

    def test_explicit_url_takes_precedence_over_env(self):
        other_url = "https://discord.com/api/webhooks/other/url"
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": other_url}):
            with patch("urllib.request.urlopen", return_value=_fake_response(204)) as mock_open:
                send_discord("explicit", webhook_url=FAKE_URL)
        req = mock_open.call_args[0][0]
        assert req.full_url == FAKE_URL


# ---------------------------------------------------------------------------
# File attachment tests
# ---------------------------------------------------------------------------

class TestSendDiscordWithFiles:
    def _write_png(self, tmp_path: Path, name: str = "plot.png") -> Path:
        p = tmp_path / name
        # minimal valid PNG header + IEND chunk (not actually displayable, but openable)
        p.write_bytes(
            b"\x89PNG\r\n\x1a\n"  # signature
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return p

    def test_multiple_files_attached(self, tmp_path):
        paths = [self._write_png(tmp_path, f"img{i}.png") for i in range(3)]
        with patch("urllib.request.urlopen", return_value=_fake_response(204)) as mock_open:
            send_discord("multi", image_paths=paths, webhook_url=FAKE_URL)
        req = mock_open.call_args[0][0]
        ct = req.get_header("Content-type")
        assert "multipart/form-data" in ct

    def test_missing_file_skipped_with_warning(self, tmp_path, capsys):
        ghost = tmp_path / "ghost.png"
        with patch("urllib.request.urlopen", return_value=_fake_response(204)):
            send_discord("ok", image_paths=[ghost], webhook_url=FAKE_URL)
        captured = capsys.readouterr()
        assert "ghost.png" in captured.out

    def test_all_files_missing_sends_text_only(self, tmp_path):
        ghost = tmp_path / "ghost.png"
        with patch("urllib.request.urlopen", return_value=_fake_response(204)) as mock_open:
            send_discord("text only", image_paths=[ghost], webhook_url=FAKE_URL)
        req = mock_open.call_args[0][0]
        # Should fall back to plain JSON (no multipart)
        assert req.get_header("Content-type") == "application/json"

    def test_files_closed_after_send(self, tmp_path):
        """Ensure files are read into memory and not left open."""
        p = self._write_png(tmp_path)
        opened_handles: list = []
        original_open = open

        def tracking_open(path, mode="r", **kwargs):
            fh = original_open(path, mode, **kwargs)
            opened_handles.append(fh)
            return fh

        with patch("builtins.open", side_effect=tracking_open):
            with patch("urllib.request.urlopen", return_value=_fake_response(204)):
                send_discord("test", image_paths=[p], webhook_url=FAKE_URL)

        for fh in opened_handles:
            assert fh.closed, f"File handle for {fh.name} was not closed"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestSendDiscordErrors:
    def test_http_error_raises_discord_error(self):
        with patch("urllib.request.urlopen", side_effect=_http_error(500, "server error")):
            with pytest.raises(DiscordError) as exc_info:
                send_discord("fail", webhook_url=FAKE_URL)
        assert exc_info.value.status == 500

    def test_rate_limit_raises_discord_error_429(self):
        with patch("urllib.request.urlopen", side_effect=_http_error(429, "rate limited")):
            with pytest.raises(DiscordError) as exc_info:
                send_discord("spam", webhook_url=FAKE_URL)
        assert exc_info.value.status == 429
        assert "429" in str(exc_info.value)

    def test_network_error_propagates(self):
        import urllib.error as ue
        with patch("urllib.request.urlopen", side_effect=ue.URLError("timeout")):
            with pytest.raises(ue.URLError):
                send_discord("net fail", webhook_url=FAKE_URL)

    def test_discord_error_attributes(self):
        with patch("urllib.request.urlopen", side_effect=_http_error(403, "forbidden")):
            with pytest.raises(DiscordError) as exc_info:
                send_discord("bad auth", webhook_url=FAKE_URL)
        err = exc_info.value
        assert err.status == 403
        assert "forbidden" in err.body

