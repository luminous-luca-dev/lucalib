# lucalib

**English** | [日本語](./README.ja.md)

A Python library for recording, plotting, and notifying training metrics in PyTorch experiments.  
Logs loss and accuracy per epoch, saves charts with Matplotlib, and sends results to Discord via Webhook — all with a clean, minimal API designed to drop straight into your training loop.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/luminous-luca-dev/lucalib.git
```

On Google Colab, run in a cell:

```python
%pip install git+https://github.com/luminous-luca-dev/lucalib.git
```

---

## Public API

| Symbol | Description |
|---|---|
| `History` | Records per-epoch loss and accuracy |
| `plot_history` | Plots training history, saves as PNG, and returns `(fig, saved_path)` |
| `send_discord` | Sends a message (and optional images) to a Discord Webhook |
| `DiscordError` | Exception raised when Discord returns a non-2xx HTTP response |

---

## Usage

### 1. Recording Training History

```python
from lucalib import History

history = History()

# Call once per epoch inside your training loop
history.append(
    train_loss=float(np.mean(losses_train)),   # numpy scalars and PyTorch tensors are accepted
    valid_loss=float(np.mean(losses_valid)),
    train_acc=train_true_num / train_num,
    valid_acc=valid_true_num / valid_num,
)

# Access recorded values at any time
best_val_acc = max(history["valid_acc"])
best_epoch   = history["valid_acc"].index(best_val_acc)
```

`History.append()` accepts `train_loss`, `valid_loss`, `train_acc`, and `valid_acc` as keyword arguments. PyTorch tensors with `.item()` are automatically converted to `float`.

---

### 2. Generating and Saving a Plot

```python
from pathlib import Path
from lucalib import plot_history

# Variable reference:
#   lr          – learning rate used in this experiment (float)
#   n_epochs    – total number of epochs (int)
#   batch_size  – batch size (int)
#   hid_dim     – hidden layer size or model description (int or str)
#   record_dir  – directory to save the chart (str / Path)
#   elapsed_time– training duration in seconds (float)
#   dropout_rate– dropout rate (float)

record_dir = Path("drive/MyDrive/Colab Notebooks/Lecture04/record")

fig, saved_path = plot_history(
    history.as_dict(),          # pass the result of History.as_dict()
    lr=lr,
    n_epochs=n_epochs,
    batch_size=batch_size,
    hidden_dims=hid_dim,
    best_val_acc=max(history["valid_acc"]),
    output_dir=record_dir,      # defaults to ../record/ relative to cwd
    comment=f"Amsgrad,DropOut:{dropout_rate},Time:{elapsed_time:.1f}s",
)

print(saved_path)   # e.g. /content/drive/.../record/exp_20260112_153045_123456.png
```

`plot_history()` returns `(fig, saved_path)`. `saved_path` is a `pathlib.Path` and can be passed directly to `send_discord`.

---

### 3. Discord Webhook Notifications

#### Setting the Webhook URL

**Using Google Colab Secrets (recommended):**

```python
from google.colab import userdata
import os

os.environ["DISCORD_WEBHOOK_URL"] = userdata.get("DISCORD_WEBHOOK_URL")
# or pass it directly via the webhook_url= argument
```

**Using an environment variable:**

```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

> [!CAUTION]
> Never hardcode your Webhook URL in source code or README files.
> Use Colab Secrets or environment variables instead.

#### Text-only message

```python
from lucalib import send_discord

send_discord("Training run 1 complete.")
# If webhook_url is omitted, the DISCORD_WEBHOOK_URL env var is used automatically.
```

#### Text + multiple images

```python
send_discord(
    "All experiments complete.",
    image_paths=["/path/to/plot1.png", "/path/to/plot2.png"],
    webhook_url="https://discord.com/api/webhooks/...",  # optional
    timeout=60,
)
```

#### Error handling

```python
from lucalib import send_discord, DiscordError
import urllib.error

try:
    send_discord("test notification", webhook_url="https://...")
except ValueError as e:
    print(f"No URL configured: {e}")
except DiscordError as e:
    print(f"Discord error (HTTP {e.status}): {e.body}")
except urllib.error.URLError as e:
    print(f"Network error: {e}")
```

---

### 4. Full Colab Example — Multi-Experiment Loop

```python
# --- Setup ---
%pip install git+https://github.com/luminous-luca-dev/lucalib.git

import os, time
import numpy as np
from pathlib import Path
from google.colab import userdata
from lucalib import History, plot_history, send_discord, DiscordError

# Load Webhook URL from Colab Secrets
os.environ["DISCORD_WEBHOOK_URL"] = userdata.get("DISCORD_WEBHOOK_URL")

# Configuration
work_dir     = "drive/MyDrive/Colab Notebooks/DLBasics2026_colab"
record_dir   = Path(work_dir) / "Lecture04/record"
n_epochs     = 60
batch_size   = 32
hid_dim      = 200

lr_list      = [0.0005, 0.001]
dropout_list = [0.3]

# --- Experiment loop ---
experiment_results = []
count = 0

for lr in lr_list:
    for dropout_rate in dropout_list:
        count += 1
        history = History()
        start_time = time.perf_counter()

        # Your existing training loop goes here — just add history.append() at the end of each epoch
        for epoch in range(n_epochs):
            # ... training code ...
            history.append(
                train_loss=np.mean(losses_train),
                valid_loss=np.mean(losses_valid),
                train_acc=train_true_num / train_num,
                valid_acc=valid_true_num / valid_num,
            )

        # --- After training: save chart ---
        elapsed_time = time.perf_counter() - start_time
        fig, saved_path = plot_history(
            history.as_dict(),
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size,
            hidden_dims=hid_dim,
            best_val_acc=max(history["valid_acc"]),
            output_dir=record_dir,
            comment=f"Amsgrad,DropOut:{dropout_rate},Time:{elapsed_time:.1f}s",
        )

        # --- Notify after each experiment ---
        send_discord(f"Run {count} finished in {elapsed_time:.1f}s.")

        experiment_results.append({
            "lr":           lr,
            "dropout":      dropout_rate,
            "best_val_acc": max(history["valid_acc"]),
            "best_epoch":   history["valid_acc"].index(max(history["valid_acc"])),
            "elapsed_time": elapsed_time,
            "png_path":     str(saved_path),
        })

# --- Final summary notification with all plots ---
message = "All experiments complete!\n\n"
for i, r in enumerate(experiment_results, start=1):
    message += (
        f"LR: `{r['lr']}` | Dropout: `{r['dropout']}` | {r['elapsed_time']:.1f}s\n"
        f"Best Valid Acc: **{r['best_val_acc']:.4f}** | Epoch: `{r['best_epoch']}`\n\n"
    )

image_paths = [r["png_path"] for r in experiment_results if Path(r["png_path"]).exists()]

send_discord(message, image_paths=image_paths)
```

---

## `send_discord` Reference

| Argument | Type | Description |
|---|---|---|
| `message` | `str` | Text content to send (required) |
| `image_paths` | `list[str \| Path] \| None` | Paths to image files to attach (optional) |
| `webhook_url` | `str \| None` | Webhook URL; falls back to `DISCORD_WEBHOOK_URL` env var if omitted |
| `timeout` | `float` | Request timeout in seconds (default: 60) |

**Returns:** `None`

**Exceptions:**

| Exception | Condition |
|---|---|
| `ValueError` | No URL provided via argument or environment variable |
| `DiscordError` | Discord returned a non-2xx status (`.status` and `.body` attributes available) |
| `urllib.error.URLError` | Network error or timeout |

Missing image paths are skipped with a warning — no exception is raised.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.9 | — |
| Matplotlib | ≥ 3.5 | Chart rendering and saving |

`send_discord` uses only the Python standard library (`urllib`, `json`, `os`, `pathlib`) — no extra packages required.

---

## Running Tests

```bash
cd lucalib
pip install -e ".[dev]"
python -m pytest tests/ -v
```

All HTTP calls are mocked — no real Webhook is needed to run the test suite.
