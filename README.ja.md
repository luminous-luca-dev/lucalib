# lucalib

[English](./README.md) | **日本語** | [🔰 初心者向け](./QUICKSTART.ja.md)

機械学習でのAccなどのログをまとめてグラフ出力できるpythonライブラリ。整った関数で簡単に呼び出し出力！
今のところは、自作の学習履歴を記録してMatplotlibで損失と精度のグラフを描画する小さめのライブラリです。
機械学習専用のplotライブラリってだけじゃ味気ないので、将来的にもっとライブラリ増やしていけるように名前に汎用性を持たせています。

Discordに通知できる関数を追加しました！

## インストール

GitHub から直接インストールできます。

```bash
pip install git+https://github.com/luminous-luca-dev/lucalib.git
```

Google Colab では次のようにセルに入力してください。

```python
%pip install git+https://github.com/luminous-luca-dev/lucalib.git
```

---

## 公開 API 一覧

| シンボル | 説明 |
|---|---|
| `History` | エポックごとの loss・accuracy を記録するクラス |
| `plot_history` | 学習履歴をグラフ化して PNG 保存し `(fig, saved_path)` を返す |
| `send_discord` | Discord Webhook にテキスト・画像を送信する |
| `DiscordError` | Discord が非 2xx を返したときに raise される例外 |

---

## 使い方

### 1. 学習履歴の記録

```python
from lucalib import History

history = History()

# エポックループ内で呼び出す
history.append(
    train_loss=float(np.mean(losses_train)),   # numpy スカラーや PyTorch テンソルも可
    valid_loss=float(np.mean(losses_valid)),
    train_acc=train_true_num / train_num,
    valid_acc=valid_true_num / valid_num,
)

# 任意のエポックの値を参照
best_val_acc = max(history["valid_acc"])
best_epoch   = history["valid_acc"].index(best_val_acc)
```

`History.append()` は `train_loss`、`valid_loss`、`train_acc`、`valid_acc` をキーワード引数で受け取ります。PyTorch テンソルの `.item()` を持つ値も自動的に `float` に変換します。

---

### 2. グラフの生成と保存

```python
from pathlib import Path
from lucalib import plot_history

# 変数の意味:
#   lr          – 実験で使った学習率 (float)
#   n_epochs    – 総エポック数 (int)
#   batch_size  – バッチサイズ (int)
#   hid_dim     – 隠れ層ユニット数など、モデル構造の説明 (int や文字列)
#   record_dir  – グラフの保存先ディレクトリ (str / Path)
#   elapsed_time– 学習にかかった秒数 (float)
#   dropout_rate– ドロップアウト率 (float)

record_dir = Path("drive/MyDrive/Colab Notebooks/Lecture04/record")

fig, saved_path = plot_history(
    history.as_dict(),          # History.as_dict() の結果を渡す
    lr=lr,
    n_epochs=n_epochs,
    batch_size=batch_size,
    hidden_dims=hid_dim,
    best_val_acc=max(history["valid_acc"]),
    output_dir=record_dir,      # 省略時は作業ディレクトリの ../record/
    comment=f"Amsgrad,DropOut:{dropout_rate},Time:{elapsed_time:.1f}s",
)

print(saved_path)   # 例: /content/drive/.../record/exp_20260112_153045_123456.png
```

`plot_history()` は `(fig, saved_path)` を返します。`saved_path` は `pathlib.Path` なので、後で `send_discord` に渡せます。

---

### 3. Discord Webhook 通知

#### Webhook URL の設定方法

**Google Colab Secrets を使う場合（推奨）:**

```python
from google.colab import userdata
import os

os.environ["DISCORD_WEBHOOK_URL"] = userdata.get("DISCORD_WEBHOOK_URL")
# または webhook_url= 引数に直接渡す
```

**環境変数を使う場合:**

```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

> [!CAUTION]
> Webhook URL はソースコードや README に直接書かないでください。
> Colab Secrets または環境変数を使用してください。

#### テキストのみ送信

```python
from lucalib import send_discord

send_discord("1回目の学習が完了しました。")
# webhook_url を省略すると DISCORD_WEBHOOK_URL 環境変数が使われます
```

#### テキスト＋複数画像を送信

```python
send_discord(
    "全実験が完了しました。",
    image_paths=["/path/to/plot1.png", "/path/to/plot2.png"],
    webhook_url="https://discord.com/api/webhooks/...",  # 省略可
    timeout=60,
)
```

#### エラーハンドリング

```python
from lucalib import send_discord, DiscordError
import urllib.error

try:
    send_discord("通知テスト", webhook_url="https://...")
except ValueError as e:
    print(f"URL 未設定: {e}")
except DiscordError as e:
    print(f"Discord エラー (HTTP {e.status}): {e.body}")
except urllib.error.URLError as e:
    print(f"通信エラー: {e}")
```

---

### 4. 複数実験ループへの組み込み例（Colab 完全例）

```python
# --- セットアップ ---
%pip install git+https://github.com/luminous-luca-dev/lucalib.git

import os, time
import numpy as np
from pathlib import Path
from google.colab import userdata
from lucalib import History, plot_history, send_discord, DiscordError

# Webhook URL を Colab Secrets から読み込む
os.environ["DISCORD_WEBHOOK_URL"] = userdata.get("DISCORD_WEBHOOK_URL")

# 変数定義
work_dir     = "drive/MyDrive/Colab Notebooks/DLBasics2026_colab"
record_dir   = Path(work_dir) / "Lecture04/record"
n_epochs     = 60
batch_size   = 32
hid_dim      = 200

lr_list      = [0.0005, 0.001]
dropout_list = [0.3]

# --- 実験ループ ---
experiment_results = []
count = 0

for lr in lr_list:
    for dropout_rate in dropout_list:
        count += 1
        history = History()
        start_time = time.perf_counter()

        # ここに既存の学習ループをそのまま書く
        # （エポックループの末尾で history.append(...) を呼ぶだけ）
        for epoch in range(n_epochs):
            # ... 学習処理 ...
            history.append(
                train_loss=np.mean(losses_train),
                valid_loss=np.mean(losses_valid),
                train_acc=train_true_num / train_num,
                valid_acc=valid_true_num / valid_num,
            )

        # --- 学習終了後: グラフ保存 ---
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

        # --- 1実験ごとの通知 ---
        send_discord(f"{count}回目の学習が {elapsed_time:.1f}s で終了しました。")

        experiment_results.append({
            "lr":           lr,
            "dropout":      dropout_rate,
            "best_val_acc": max(history["valid_acc"]),
            "best_epoch":   history["valid_acc"].index(max(history["valid_acc"])),
            "elapsed_time": elapsed_time,
            "png_path":     str(saved_path),
        })

# --- 全実験終了後: 結果まとめ通知 ---
message = "ますたー、全実験完了しました～\n\n"
for i, r in enumerate(experiment_results, start=1):
    message += (
        f"LR: `{r['lr']}` | Dropout: `{r['dropout']}` | {r['elapsed_time']:.1f}s\n"
        f"Best Valid Acc: **{r['best_val_acc']:.4f}** | Epoch: `{r['best_epoch']}`\n\n"
    )

image_paths = [r["png_path"] for r in experiment_results if Path(r["png_path"]).exists()]

send_discord(message, image_paths=image_paths)
```

---

## `send_discord` の引数・返り値

| 引数 | 型 | 説明 |
|---|---|---|
| `message` | `str` | 送信するテキスト（必須） |
| `image_paths` | `list[str \| Path] \| None` | 添付する画像のパス一覧（省略可） |
| `webhook_url` | `str \| None` | Webhook URL。省略時は `DISCORD_WEBHOOK_URL` 環境変数を使用 |
| `timeout` | `float` | タイムアウト秒数（デフォルト 60） |

**返り値:** なし (`None`)

**例外:**

| 例外 | 条件 |
|---|---|
| `ValueError` | URL が引数にも環境変数にも未設定 |
| `DiscordError` | Discord が非 2xx を返した（`.status` と `.body` 属性あり） |
| `urllib.error.URLError` | ネットワークエラー・タイムアウト |

存在しない画像パスは個別に警告を出してスキップします（例外は raise されません）。

---

## 依存関係

| パッケージ | バージョン | 用途 |
|---|---|---|
| Python | ≥ 3.9 | — |
| Matplotlib | ≥ 3.5 | グラフ描画・保存 |

`send_discord` は `urllib`・`json`・`os`・`pathlib` のみ使用し、追加の外部ライブラリは不要です。

---

## テストの実行

```bash
# リポジトリのルートから
cd lucalib
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Webhook への実際の HTTP 通信はモックで置き換えられるため、ネットワーク不要で全テストが実行できます。

