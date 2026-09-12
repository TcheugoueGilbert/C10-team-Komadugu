# data/

This project does not bundle the Civil Comments dataset itself (it is
~2M rows and is redistributed by Hugging Face under its own license).
Instead, this folder holds artifacts *generated* by the pipeline:

- `embeddings/` — Gemma-2-2B layer-14 mean-pooled embeddings saved by
  `scripts/pipeline/01_extract_embeddings.py`
  (`X_train_layer14.npy`, `y_train.npy`, `X_val_layer14.npy`, `y_val.npy`)
- `models/` — the trained linear probe and hyperparameter search log
  saved by `scripts/pipeline/02_train_probe.py`
  (`classifier.joblib`, `hyperparameter_search.csv`)

Both subfolders are populated automatically the first time you run the
pipeline (see the root `README.md` for the exact commands) and are
excluded from version control via `.gitignore`.

To fetch/verify the raw dataset itself (cached by Hugging Face, not
stored here), run:

```bash
python scripts/pipeline/00_download_data.py
```
