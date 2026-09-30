# IEEE-CIS Fraud Detection (Kaggle)

Raw files are **not** committed. Download with:

```bash
make sync
make download-data
```

This uses [`kagglehub`](https://github.com/Kaggle/kagglehub):

```python
import kagglehub
path = kagglehub.competition_download("ieee-fraud-detection")
```

Configure Kaggle credentials per [Kaggle API docs](https://github.com/Kaggle/kaggle-api):

- `~/.kaggle/kaggle.json` with `"username"` and `"key"`, or
- environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`

Files are copied into `data/raw/`:

- `train_transaction.csv`
- `train_identity.csv`

License: see the competition page on Kaggle.
