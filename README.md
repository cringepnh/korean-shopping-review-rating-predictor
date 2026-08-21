# Korean Shopping Review Rating Predictor

Predict one of the real ratings **1, 2, 4, or 5** from Korean Naver Shopping
review text. This is a shopping-review project, not a movie-review project.

The previous version was invalid: it mapped NSMC binary sentiment labels to
random pseudo-ratings and misdescribed ordinary cross-entropy as order-aware. Those data,
fallbacks, metrics, plots, and custom model artifacts have been removed. No
target is generated from sentiment or from an RNG in this version.

## Why this replacement dataset

Path A was chosen because the
[`bab2min/corpus`](https://github.com/bab2min/corpus) Naver Shopping file is
currently downloadable, contains ratings in the source file, and its upstream
repository declares the corpus **Public Domain**. Scraping movie sites (Path B)
would add changing endpoints and unclear robots/terms risk. Deleting the project
(Path C) was unnecessary once an auditable real-rating source was available.

Source file:
[`sentiment/naver_shopping.txt`](https://github.com/bab2min/corpus/blob/master/sentiment/naver_shopping.txt)

Pinned SHA-256:
`dc4d1aca0a148671cbe80bcb81962eee297370acab42be93c1617ce9336479c0`

`download_data.py` raises and stops on HTTP errors or a hash mismatch. It has
no synthetic or NSMC fallback. Training also verifies the hash, the manifest,
the split schema, and `target_origin == "source_file"` on every row.

## Data

The downloaded file has 200,000 rows. All 184 rows belonging to 92 identical
texts with conflicting source ratings are removed rather than arbitrarily
choosing one label. The resulting 199,816 unique, unambiguous reviews are split
80/10/10 with seed 42 and no exact text overlap.

| Rating | Clean rows | Share |
|---:|---:|---:|
| 1 | 35,970 | 18.00% |
| 2 | 63,903 | 31.98% |
| 3 | **0** | **0.00%** |
| 4 | 18,776 | 9.40% |
| 5 | 81,167 | 40.62% |

Every data load prints the target distribution and a loud warning that rating
3 has zero examples. The model therefore cannot learn or emit a 3-star class.

| Split | Rows |
|---|---:|
| Train | 159,852 |
| Validation | 19,982 |
| Test | 19,982 |

## Method and evaluation results

The model is standard **4-class classification** over `[1, 2, 4, 5]` using
`AutoModelForSequenceClassification` and cross-entropy. No order-aware loss or
threshold formulation is implemented. The
broken regression model was removed rather than presented as a comparison.

Every row below is evaluated on the same held-out 19,982-review test split.

| System | Prediction | MAE | RMSE | Exact accuracy | Within ±1 |
|---|---:|---:|---:|---:|---:|
| Constant: training median | 4.0000 | 1.5859 | 1.8182 | 9.39% | 50.02% |
| Constant: training mean | 3.2265 | 1.5862 | 1.6454 | 0.00% | 9.39% |
| Majority rating class | 5 | 1.7735 | 2.4192 | 40.62% | 50.02% |
| Fine-tuned 4-class KoELECTRA | — | **0.3856** | **0.8364** | **71.66%** | **94.46%** |

The constant mean is allowed to be non-integer for MAE/RMSE, hence its exact
accuracy is zero. The model improves MAE by 1.2003 relative to the strongest
constant-MAE baseline (75.7% relative reduction). Epoch 2 was selected using
validation MAE 0.3894; epoch 3 was slightly worse and was not used. Full-precision
metrics and validation history are in `evaluation_results.json`.

## Reproduce

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python download_data.py
python prepare_data.py
python train_model.py --baselines-only
python train_model.py --epochs 3 --train-batch-size 32 --eval-batch-size 128
python evaluate_model.py
```

The raw base checkpoint is
[`monologg/koelectra-base-v3-discriminator`](https://huggingface.co/monologg/koelectra-base-v3-discriminator).
The best checkpoint is selected only by validation MAE; test is evaluated after
training. A standard Transformers export is saved under
`models/rating-classifier/`.

```bash
python predict.py "배송도 빠르고 제품도 정말 좋아요"
```

`predict.py` displays probabilities only for ratings 1, 2, 4, and 5.

## Project files

- `download_data.py` — real-source download, fixed-hash verification, manifest
- `data_utils.py` — provenance assertions, distributions, and metrics
- `prepare_data.py` — deterministic, disjoint 80/10/10 splits
- `train_model.py` — baselines plus 4-class KoELECTRA fine-tuning
- `evaluate_model.py` — saved-model and same-test baseline comparison
- `predict.py` — inference from the standard Transformers export

## Limitations

- The domain is Naver Shopping product reviews, not movies.
- The upstream corpus excludes 3-star reviews. Predictions are not a complete
  1–5 rating scale.
- The class distribution is imbalanced, especially for rating 4.
- Text alone may not contain enough information to recover a user's exact
  numeric rating.
- Metrics from one split and seed are not uncertainty estimates.
- The local folder was renamed, but the remote GitHub repository must be
  renamed separately by its owner before publication.

Project code is MIT-licensed. Dataset licensing is separate: the upstream
`bab2min/corpus` repository declares the corpus Public Domain. Users should
review the upstream source for their own use case.
