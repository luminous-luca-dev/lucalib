# lucalib
機械学習でのAccなどのログをまとめてグラフ出力できるpythonライブラリ。整った関数で簡単に呼び出し出力！

Matplotlib helper for plotting train/validation loss and accuracy from a
`history` dictionary. It keeps the original 12 × 7 figure, twin axes, colors,
line styles, combined legend, and grid.

## Install locally

In a terminal, move into this folder and run:

```bash
python -m pip install .
```

For an editable install while developing the package:

```bash
python -m pip install -e .
```

In Google Colab, upload the package folder to Drive, then run (adjust the path):

```python
%pip install /content/drive/MyDrive/path/to/lucalib
```

## Use

```python
from lucalib import History, plot_history

history = History()

fig, saved_path = plot_history(
    history.as_dict(),
    lr=lr,
    n_epochs=n_epochs,
    batch_size=batch_size,
    hidden_dims=hidden_dims,
    best_val_acc=best_val_acc,
    comment="Add He, Adam",
)
print(saved_path)
```

After each epoch, add one call to record the epoch's mean metrics:

```python
history.append(
    train_loss=np.mean(losses_train),
    valid_loss=np.mean(losses_valid),
    train_acc=train_true_num / train_num,
    valid_acc=valid_true_num / valid_num,
)
```

The four required history keys are `train_loss`, `valid_loss`, `train_acc`,
and `valid_acc`. `History.append` takes care of accumulating one value per
epoch. The plot is saved automatically to `../record` relative to the current
working directory. Pass `output_dir="some/folder"` to choose another location;
`show=False` suppresses display. The function returns `(figure, saved_path)`.
Close the figure after use in long-running scripts with `plt.close(fig)`.
