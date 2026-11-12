# lucalib クイックスタート（初心者向け）

> このガイドは **Google Colab** での利用を想定しています。
> コードブロックは基本的に**丸ごとコピーして貼り付けるだけ**で動きます。
> 変更が必要な箇所には `# ← ここだけ変える` と書いてあります。

---

## 目次

1. [インストール](#1-インストール)
2. [セットアップ（最初に1回だけ実行）](#2-セットアップ最初に1回だけ実行)
3. [学習ループへの組み込み](#3-学習ループへの組み込み)
4. [全実験後のまとめ送信](#4-全実験後のまとめ送信)

---

## 1. インストール

新しいセルに以下をコピーして実行してください。

```python
%pip install git+https://github.com/luminous-luca-dev/lucalib.git
```

---

## 2. セットアップ（最初に1回だけ実行）

以下のセルを**丸ごとコピー**してください。

**変更が必要なのは `RECORD_DIR` の1行だけです。**  
グラフの保存先フォルダのパスを自分の環境に合わせて書き換えてください。

```python
# ============================================================
#  lucalib セットアップセル（丸ごとコピーしてください）
# ============================================================
import os
import time
import numpy as np
from pathlib import Path
from google.colab import userdata
from lucalib import History, plot_history, send_discord

# ↓ グラフの保存先フォルダを自分のパスに変えてください
RECORD_DIR = Path("drive/MyDrive/Colab Notebooks/record")  # ← ここだけ変える

# Discord Webhook URL の読み込み
# （Colab の左サイドバー「🔑 シークレット」に DISCORD_WEBHOOK_URL という名前で登録してください）
os.environ["DISCORD_WEBHOOK_URL"] = userdata.get("DISCORD_WEBHOOK_URL")


# ------------------------------------------------------------
#  以下はコピーするだけでOK。変更不要です。
# ------------------------------------------------------------

def finish_experiment(
    history,
    *,
    lr,
    n_epochs,
    batch_size,
    hidden_dims,
    dropout_rate,
    elapsed_time,
    count,
    experiment_results,
    comment_extra="",
):
    """
    1回の実験が終わったら呼び出す関数。
    グラフの保存・Discord通知・結果リストへの追記をまとめてやってくれます。
    """
    comment = f"DropOut:{dropout_rate}, Time:{elapsed_time:.1f}s"
    if comment_extra:
        comment += f", {comment_extra}"

    fig, saved_path = plot_history(
        history.as_dict(),
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        hidden_dims=hidden_dims,
        best_val_acc=max(history["valid_acc"]),
        output_dir=RECORD_DIR,
        comment=comment,
        show=False,
    )

    send_discord(f"{count}回目の学習が {elapsed_time:.1f}s で終了しました。")

    experiment_results.append({
        "lr":           lr,
        "dropout":      dropout_rate,
        "best_val_acc": max(history["valid_acc"]),
        "best_epoch":   history["valid_acc"].index(max(history["valid_acc"])),
        "elapsed_time": elapsed_time,
        "png_path":     str(saved_path),
    })


def send_summary(experiment_results):
    """
    全実験が終わったら呼び出す関数。
    結果の一覧とグラフ画像をまとめて Discord に送ります。
    """
    message = "全実験完了！\n\n"
    for r in experiment_results:
        message += (
            f"LR: `{r['lr']}` | Dropout: `{r['dropout']}` | {r['elapsed_time']:.1f}s\n"
            f"Best Valid Acc: **{r['best_val_acc']:.4f}** | Epoch: `{r['best_epoch']}`\n\n"
        )
    image_paths = [r["png_path"] for r in experiment_results if Path(r["png_path"]).exists()]
    send_discord(message, image_paths=image_paths)

print("セットアップ完了！")
```

> [!NOTE]
> `DISCORD_WEBHOOK_URL` の登録方法：  
> Colab 画面左のサイドバーにある 🔑（シークレット）アイコン → 「新しいシークレットを追加」→  
> 名前: `DISCORD_WEBHOOK_URL`、値: Discord の Webhook URL を貼り付ける

---

## 3. 学習ループへの組み込み

既存の学習コードに **3か所だけ** 追記します。

```python
# ============================================================
#  実験ループ（変更が必要な箇所にコメントをつけています）
# ============================================================

# ① 実験ループの前に追加（コピーするだけでOK）
experiment_results = []
count = 0

# ↓ 試したいパラメータのリストを書いてください
lr_list      = [0.001]        # ← 変える
dropout_list = [0.3]          # ← 変える

for lr in lr_list:
    for dropout_rate in dropout_list:
        count += 1

        # ② エポックループの前に追加（コピーするだけでOK）
        history = History()
        start_time = time.perf_counter()

        # ===== ここに既存の学習ループをそのまま書く =====
        for epoch in range(n_epochs):

            # ... 自分の学習コード（変更不要） ...

            # ③ エポックの末尾に追加（変数名だけ自分のコードに合わせる）
            history.append(
                train_loss=np.mean(losses_train),       # ← 変数名を合わせる
                valid_loss=np.mean(losses_valid),       # ← 変数名を合わせる
                train_acc=train_true_num / train_num,   # ← 変数名を合わせる
                valid_acc=valid_true_num / valid_num,   # ← 変数名を合わせる
            )
        # ===== 学習ループここまで =====

        # ④ 実験終了後に追加（コピーするだけでOK）
        elapsed_time = time.perf_counter() - start_time
        finish_experiment(
            history,
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size,
            hidden_dims=hid_dim,
            dropout_rate=dropout_rate,
            elapsed_time=elapsed_time,
            count=count,
            experiment_results=experiment_results,
        )
```

> [!TIP]
> `③ history.append()` の変数名は自分のコードに合わせて変えてください。  
> たとえば `losses_train` を `loss_list` という名前にしているなら `np.mean(loss_list)` と書きます。  
> それ以外はコピーするだけでOKです。

---

## 4. 全実験後のまとめ送信

実験ループの直後に以下を追加するだけです。

```python
# ============================================================
#  全実験完了後（コピーするだけでOK）
# ============================================================
send_summary(experiment_results)
```

これだけで、全実験の結果一覧と保存したグラフ画像がまとめて Discord に届きます。

---

## まとめ：自分で書く部分はここだけ

| 箇所 | 変更内容 |
|---|---|
| `RECORD_DIR` | グラフの保存先パス |
| `lr_list` / `dropout_list` | 試したいパラメータ |
| `history.append()` の変数名 | `losses_train` などを自分のコードの変数名に合わせる |
| 既存の学習コード | そのまま変更なし |

---

## うまくいかないときは

| 症状 | 確認するポイント |
|---|---|
| `ValueError: Discord Webhook URL が設定されていません` | Colab のシークレットに `DISCORD_WEBHOOK_URL` が登録されているか確認 |
| `ModuleNotFoundError: No module named 'lucalib'` | インストールのセルを実行したか確認。ランタイムを再起動してから再実行 |
| グラフが保存されない | `RECORD_DIR` のパスが正しいか、Google Drive がマウントされているか確認 |
| `DiscordError: HTTP 429` | しばらく待ってから再実行（Discord の送信制限に引っかかっている） |

