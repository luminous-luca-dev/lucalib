"""Discord Webhook notification helpers.

Send a text message, optionally with multiple image attachments, to a Discord
Webhook URL.  No third-party libraries are required beyond the standard library;
network requests use ``urllib`` which is part of the Python standard library.

If you prefer to keep the Webhook URL out of your source code you can store it
in the environment variable ``DISCORD_WEBHOOK_URL``.  The helper reads that
variable automatically when no *webhook_url* argument is supplied.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Multipart helpers (pure stdlib – no requests dependency)
# ---------------------------------------------------------------------------

def _build_multipart(
    payload_json: str,
    files: list[tuple[str, str, bytes]],  # (field_name, filename, data)
) -> tuple[bytes, str]:
    """Return (body_bytes, content_type_header_value) for a multipart/form-data request."""
    boundary = "----LucalibBoundary7MA4YWxkTrZu0gW"
    parts: list[bytes] = []
    crlf = b"\r\n"

    def _part(name: str, value: str | bytes, filename: str | None = None, mime: str = "text/plain") -> bytes:
        lines: list[bytes] = []
        cd = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            cd += f'; filename="{filename}"'
        lines.append(f"--{boundary}".encode())
        lines.append(cd.encode())
        if filename:
            lines.append(f"Content-Type: {mime}".encode())
        lines.append(b"")
        lines.append(value if isinstance(value, bytes) else value.encode())
        return crlf.join(lines)

    parts.append(_part("payload_json", payload_json))
    for field_name, filename, data in files:
        parts.append(_part(field_name, data, filename=filename, mime="image/png"))

    body = crlf.join(parts) + crlf + f"--{boundary}--".encode() + crlf
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DiscordError(RuntimeError):
    """Raised when Discord returns a non-2xx HTTP response."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"Discord Webhook returned HTTP {status}: {body}")


def send_discord(
    message: str,
    image_paths: Sequence[str | Path] | None = None,
    *,
    webhook_url: str | None = None,
    timeout: float = 60.0,
) -> None:
    """Send *message* to a Discord Webhook, optionally attaching image files.

    Parameters
    ----------
    message:
        The text content to post.
    image_paths:
        Optional list of paths to PNG (or other image) files to attach.
        Files that do not exist are skipped with a printed warning.
    webhook_url:
        The Discord Webhook URL.  When omitted the value of the environment
        variable ``DISCORD_WEBHOOK_URL`` is used.
    timeout:
        Request timeout in seconds (default 60).

    Raises
    ------
    ValueError
        If no Webhook URL is available (neither argument nor environment variable).
    FileNotFoundError
        If *all* provided image paths are missing and there is nothing to send.
        (Individual missing files only produce a warning.)
    DiscordError
        If Discord returns a non-2xx HTTP status code.
    urllib.error.URLError
        If a network or timeout error occurs.
    """
    url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        raise ValueError(
            "Discord Webhook URLが設定されていません。"
            " webhook_url 引数か環境変数 DISCORD_WEBHOOK_URL を設定してください。"
        )

    payload_json = json.dumps({"content": message})

    # --- Collect image data while guaranteeing files are closed ---
    file_tuples: list[tuple[str, str, bytes]] = []
    for i, raw_path in enumerate(image_paths or []):
        path = Path(raw_path)
        if not path.exists():
            print(f"[lucalib] 画像が見つかりません（スキップ）: {path}")
            continue
        with open(path, "rb") as fh:
            data = fh.read()
        file_tuples.append((f"files[{i}]", path.name, data))

    # --- Build request ---
    if file_tuples:
        body, content_type = _build_multipart(payload_json, file_tuples)
    else:
        body = payload_json.encode()
        content_type = "application/json"

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )

    # --- Send and handle errors ---
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        # Discord rate-limit
        if exc.code == 429:
            raise DiscordError(
                429,
                f"レート制限に達しました (429 Too Many Requests). 詳細: {body_text}",
            ) from exc
        raise DiscordError(exc.code, body_text) from exc

    print(f"[lucalib] Discordへの送信に成功しました (HTTP {status})")

