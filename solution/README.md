# Full-text MiniLM + LightGBM solution

This folder contains the selected single-model solution. It uses one LightGBM classifier with tabular features, 384-dimensional MiniLM embeddings of the full post, and author-relative text features computed from earlier posts. It does not use an ensemble.

Reference results from the fixed experiment:

- Validation Macro-F1: **0.6406**
- Official test Macro-F1: **0.6305**
- Official test QWK: **0.5558**
- Official test accuracy: **0.6754**

## Place and install

Place this `solution/` directory in the root of the original `kairos-dataset` repository, alongside `kairos/`, `data/`, and the repository's `requirements.txt`. The release must already be unpacked in `data/`. The scripts read post text and author profiles only; they do not open media files.

Use Python 3.13. Install the original project dependencies and then the additional model dependencies:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r solution/requirements.txt
```

The first training run downloads the pinned `sentence-transformers/all-MiniLM-L6-v2` model files from Hugging Face. These are pretrained model weights, not an additional dataset. The weights are cached under `results/solution/encoder/`; subsequent runs reuse them.

## Train and calculate metrics

Run from the repository root:

```bash
.venv/bin/python solution/run_solution.py train
```

The script builds features from the local release, encodes every labelled post, trains one LightGBM model on official training rows, predicts official test rows, and calculates the test metrics. A post is split into non-overlapping pieces of up to 254 text tokens; MiniLM adds its two special tokens, so each encoder input is at most 256 tokens. Chunk vectors are averaged with token-count weights and the resulting post vector is L2-normalized.

MiniLM encoding runs in a separate Python process so that PyTorch and LightGBM do not load conflicting OpenMP runtimes into the same process on macOS. Embedding checkpoints are saved every 128 posts and reused after a restart. Keep `encoder_worker.py` alongside `run_solution.py`.

Generated files are written to `results/solution/`:

- `test_predictions.csv` with columns `author`, `post`, `label`.
- `training_report.json` with the model configuration, metrics, and timings.
- `lightgbm_model.txt` and `preprocessing.joblib` for the fitted model.
- `cache/` with reusable feature matrices and full-text embeddings.

To score the prediction CSV using the original evaluator:

```bash
.venv/bin/python evaluate.py --predictions results/solution/test_predictions.csv
```

## Recreate predictions

After training, regenerate predictions and metrics from the saved model and cache with:

```bash
.venv/bin/python solution/run_solution.py predict
```

This writes `test_predictions.csv`, `test_probabilities.npy`, and `prediction_metrics.json` under `results/solution/`.

Use a different release or output directory with `--data` and `--output`:

```bash
.venv/bin/python solution/run_solution.py train --data /path/to/data --output /path/to/output
```

A completed output folder is not overwritten by default. Use a new output path for a fresh run, or pass `--force` to retrain into the same folder. Use `--force-features` to rebuild the cached feature table and author-history matrix.
