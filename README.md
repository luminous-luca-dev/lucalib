# lucalib
機械学習でのAccなどのログをまとめてグラフ出力できるpythonライブラリ。整った関数で簡単に呼び出し出力！
今のところは、自作の学習履歴を記録してMatplotlibで損失と精度のグラフを描画する小さめのライブラリです。
機械学習専用のplotライブラリってだけじゃ味気ないので、将来的にもっとライブラリ増やしていけるように名前に汎用性を持たせています。

## インストール

GitHubから直接インストールできます。

    pip install git+https://github.com/luminous-luca-dev/lucalib.git

Google Colabでは、セルに次のように入力してください。

    %pip install git+https://github.com/luminous-luca-dev/lucalib.git

## 使い方

    import numpy as np
    from lucalib import History, plot_history

    history = History()

    # 各エポックの終了時に、平均損失と精度を記録
    history.append(
        train_loss=np.mean(losses_train),
        valid_loss=np.mean(losses_valid),
        train_acc=train_true_num / train_num,
        valid_acc=valid_true_num / valid_num,
    )

    # 学習終了後にグラフを描画・保存
    fig, saved_path = plot_history(
        history.as_dict(),
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        hidden_dims=hid_dim,
        best_val_acc=max(history["valid_acc"]),
    )

    print(saved_path)

`History.append()` には、1エポック分の `train_loss`、`valid_loss`、`train_acc`、`valid_acc` を渡します。

グラフは既定で、現在の作業ディレクトリの1つ上にある `record` フォルダーへPNG形式で保存されます。保存先を変える場合は `output_dir` を指定してください。

    fig, saved_path = plot_history(
        history.as_dict(),
        output_dir="/content/drive/MyDrive/Colab Notebooks/record",
    )

`plot_history()` は `(fig, saved_path)` を返します。`fig` はMatplotlibの図、`saved_path` は保存された画像のパスです。

## 依存関係

- Python 3.9以上
- Matplotlib 3.5以上

Matplotlibはインストール時に依存パッケージとしてインストールされます。
