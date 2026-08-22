# Korean Shopping Review Rating Predictor

Predict one of the real ratings **1, 2, 4, or 5** from Korean Naver Shopping
review text. This is a shopping-review project, not a movie-review project.

The previous version was invalid: it mapped NSMC binary sentiment labels to
random pseudo-ratings and misdescribed ordinary cross-entropy as order-aware.
Those data, fallbacks, metrics, plots, and custom model artifacts have been
removed. No target is generated from sentiment or from an RNG in this
version.

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

**These class priors are curated, not natural.** The upstream corpus was built
for sentiment experiments, and its author states both design choices
explicitly: 3-star reviews were dropped as too ambiguous to label
positive/negative, and the remainder was sampled so that positive (4–5) and
negative (1–2) sit close to a 1:1 ratio. The 200,000 source rows are 99,963
positive and 100,037 negative; after cleaning, 99,943 and 99,873. Real Naver
Shopping ratings are heavily skewed toward 5 stars, so **nothing in this
project's class distribution should be read as an estimate of how Korean
shoppers actually rate products.** Every number here describes performance on
this curated distribution. See [Limitations](#limitations).

| Split | Rows |
|---|---:|
| Train | 159,852 |
| Validation | 19,982 |
| Test | 19,982 |

## Method and evaluation results

The primary model is standard **4-class classification** over `[1, 2, 4, 5]`
using `AutoModelForSequenceClassification` and cross-entropy. A second model,
**CORAL ordinal regression** (Cao, Mirjalili & Raschka 2020), was trained to
test whether treating the ratings as genuinely ordered — rather than four
unrelated classes — improves on that. It doesn't; see
[CORAL: an ordinal-regression comparison](#coral-an-ordinal-regression-comparison)
for the full result and why.

All rows below are evaluated on the same held-out 19,982-review test split,
with 95% bootstrap confidence intervals (1,000 resamples) and quadratic
weighted kappa (QWK, computed on rank indices {0,1,2,3} — see the note below
the table).

| System | MAE | RMSE | Exact accuracy | Polarity acc.¹ | QWK |
|---|---:|---:|---:|---:|---:|
| Constant: training median | 1.5859 | 1.8182 | 9.39% | 50.02% | 0.000 |
| Constant: training mean | 1.5862 | 1.6454 | 0.00% | 9.39% | 0.000 |
| Majority rating class | 1.7735 | 2.4192 | 40.62% | 50.02% | 0.000 |
| CE + argmax | 0.3856 [0.376, 0.396] | 0.8364 | **71.66%** [71.0, 72.3] | 94.46% | 0.837 [0.831, 0.843] |
| CE + expected value | 0.4862 [0.479, 0.494] | **0.7370** | 0.01%² | 91.88% | 0.839 |
| CE + expected value, rounded | 0.4000 [0.391, 0.410] | 0.7992 | 68.53% [67.9, 69.2] | 94.52% | 0.839 |
| CE + median | 0.3858 [0.376, 0.396] | 0.8155 | 70.88% [70.2, 71.5] | 94.48% | **0.842** [0.837, 0.847] |
| CORAL ordinal regression | 0.5632 [0.553, 0.575] | 1.0051 | 56.89% [56.2, 57.5] | 94.17% | 0.810 [0.805, 0.816] |

¹ "Polarity acc." replaces the earlier "within ±1" label: on this rating
scale {1,2,4,5} with no 3-star class, `|pred-true|<=1` holds **exactly** for
the pairs {1,2}×{1,2} and {4,5}×{4,5} — i.e. it is numerically identical to
"predicted and true rating landed on the same side of the 2/4 gap." It is not
a genuine ordinal-distance metric here; QWK is. This identity is locked in by
a regression test (`tests/test_data_utils.py`), not just asserted in prose.

² The continuous expected-value decoder is never exactly one of the four
observed ratings, so its exact accuracy is ~0% by construction — the same
reason the "constant: training mean" baseline has 0% exact accuracy. Read
that row as an RMSE-only comparison point.

Best checkpoint (CE): epoch 2, validation MAE 0.3894; epoch 3 was slightly
worse. `evaluation_results.json` and `coral_results.json` are the
machine-readable sources for every number above, produced end to end by
`evaluate_model.py` — nothing here is typed in by hand.

### Four decoders, one trained classifier

All four CE rows above come from the **same** trained model's output
probabilities — no retraining, just different rules for turning a probability
distribution into a single rating (`decoding.py`). This project initially
assumed the expected value (the distribution's mean) would minimize MAE. It
doesn't: **the mean minimizes squared error (RMSE); the median minimizes
absolute error (MAE)** — a standard statistical fact this project got wrong
before checking it empirically. The table confirms the theory:

- expected value **improves RMSE** over argmax (0.7370 vs 0.8364) but **makes
  MAE worse** (0.4862 vs 0.3856);
- the median decoder is the one that actually targets MAE, and it does — but
  only marginally (0.3858 vs argmax's 0.3856): on this model's predictions,
  argmax and the discrete median rarely disagree, so the "correct" decoder
  for MAE barely moves the number in practice. Argmax remains the better
  default here because it also has the highest exact accuracy.

### Per-class breakdown (CE + argmax)

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 1 | 61.34% | 41.20% | 49.29% | 3,597 |
| 2 | 67.13% | 78.85% | 72.52% | 6,391 |
| 4 | 39.32% | **13.53%** | 20.13% | 1,877 |
| 5 | 80.15% | 92.95% | 86.08% | 8,117 |

Rating 4 (9.4% of the data, sandwiched between the much larger 2 and 5
classes) has by far the worst recall: the model rarely predicts a 4 at all.
See the [Limitations](#limitations) section — this is a real weakness, not
hidden behind the aggregate MAE.

## CORAL: an ordinal-regression comparison

A [CORAL](https://arxiv.org/abs/1901.07884)-style ordinal head (`coral.py`)
replaces the 4-way softmax with a single shared weight vector and three
learned bias terms, predicting cumulative probabilities P(rank > k) instead of
per-class probabilities. It is a genuine implementation, not a relabeling of
the same classifier.

**On the rank-monotonicity claim.** Sharing one projection across all three
thresholds means the ordering of the cumulative logits depends only on the
ordering of the three biases — the same for every input — rather than being a
per-input property that independent-weight threshold classifiers can violate
example by example. This implementation does **not** reparameterize the biases
to force descending order, so monotonicity is not architecturally guaranteed;
it is checked after training and reported either way. In this run the biases
came out monotonically non-increasing (`[0.1245, -0.1106, -0.1243]`), so the
rank-consistency property does hold here — verified, not assumed.

**Result: CORAL is worse on every metric** — MAE 0.5632 vs. CE's 0.3856 (a
paired-bootstrap 95% CI on the difference is [-0.187, -0.168], excluding
zero; an exact McNemar test on row-level correctness gives p < 0.001). The
CI on the difference is the informative statistic here — the exact p-value
is far below any reporting threshold and is kept only in
`evaluation_results.json`. This is reported as-is rather than tuned away,
because it's the more interesting result for understanding the model class
than a tie or a small win would have been.

**Why it loses:** CORAL's per-class recall collapses almost entirely to
ratings 1 and 5:

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 1 | 35.93% | 94.41% | 52.05% | 3,597 |
| 2 | 43.28% | **1.36%** | 2.64% | 6,391 |
| 4 | 10.00% | **0.05%** | 0.11% | 1,877 |
| 5 | 76.40% | 97.13% | 85.52% | 8,117 |

A single shared projection direction plus three biases is enough to separate
"clearly negative" from "clearly positive," but not enough to also carve out
the boundary between {1,2} and between {4,5} — every rating effectively
collapses toward the nearest extreme. The 4-class softmax, with an
independent decision boundary per class, doesn't have that constraint. In
short: on a 4-way scale with one deeply lopsided class (4 is 9.4% of the
data) and a real gap in the middle (no 3-star reviews), giving up per-class
flexibility for ordinal structure cost more than the ordinal structure
bought back — the opposite of what CORAL is designed to win at on a densely,
evenly populated ordinal scale.

Training took 1,674 seconds (~28 min) on an RTX 4080 Laptop GPU for 3 epochs;
best checkpoint was epoch 2 (validation MAE 0.5623). The CORAL model is
**not** published to the Hugging Face Hub (it needs `coral.py`'s custom
architecture to load, whereas the CE model loads with stock
`AutoModelForSequenceClassification`); its weights and full metrics live in
this repository (`models/rating-coral/`, `coral_results.json`).

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
python train_coral.py --smoke-test   # verify the CORAL pipeline on a tiny slice first
python train_coral.py --epochs 3 --train-batch-size 32 --eval-batch-size 128
python evaluate_model.py             # regenerates evaluation_results.json end to end
```

For unit tests (no torch needed — data cleaning, decoders, CORAL rank
encoding, bootstrap/McNemar helpers):

```bash
pip install -r requirements-dev.txt
pytest -q
```

The raw base checkpoint is
[`monologg/koelectra-base-v3-discriminator`](https://huggingface.co/monologg/koelectra-base-v3-discriminator).
The best checkpoint is selected only by validation MAE; test is evaluated
after training. A standard Transformers export is saved under
`models/rating-classifier/`.

```bash
python predict.py "배송도 빠르고 제품도 정말 좋아요"
python predict.py "배송도 빠르고 제품도 정말 좋아요" --decoder median
```

`predict.py` supports `--decoder {argmax,expected,median}` and always
restricts output to ratings 1, 2, 4, and 5.

## Try it without installing anything

A Gradio demo is in `space/` (loads the published Hub model, not a local
path), with a switch between the argmax and expected-value decoders — see
the live Space linked from the
[model card](https://huggingface.co/cringepnh/koelectra-korean-shopping-rating).

## Project files

- `download_data.py` — real-source download, fixed-hash verification, manifest
- `data_utils.py` — provenance assertions, distributions, metrics, split-leakage check
- `prepare_data.py` — deterministic, disjoint 80/10/10 splits
- `train_model.py` — baselines plus 4-class KoELECTRA fine-tuning
- `coral.py` / `ordinal_utils.py` — CORAL ordinal-regression head (the latter
  is pure numpy, split out specifically to be unit-testable without torch)
- `train_coral.py` — CORAL training and evaluation
- `decoding.py` — argmax / expected-value / median decoders over one model's probabilities
- `evaluate_model.py` — regenerates `evaluation_results.json` end to end: all
  baselines, all decoders, CORAL comparison, CIs, QWK, per-class, confusion,
  significance tests
- `reporting.py` — bootstrap CI / McNemar / per-class helpers (no torch import)
- `predict.py` — inference from the standard Transformers export
- `upload_to_hub.py` — verified model/card upload with hidden token input
- `tests/` — unit tests for the above (`pytest -q`, no GPU or data download needed)

## Publish the trained model

After revoking any old exposed token, create a new Hugging Face token with
write access and run:

```bash
hf auth login
python upload_to_hub.py
```

The official CLI stores the token in the local Hugging Face credential cache,
not in this repository. The script refuses to upload unless the base
checkpoint, source hash, rating classes, metrics file, model export, and model
card are present and consistent. Its destination is
`cringepnh/koelectra-korean-shopping-rating` — the CE model, not CORAL (see
above for why).

## Limitations

- The domain is Naver Shopping product reviews, not movies.
- The upstream corpus excludes 3-star reviews. Predictions are not a complete
  1–5 rating scale.
- **The corpus was sampled to keep positive (4–5) and negative (1–2) reviews
  close to a 1:1 ratio, so its class priors do not represent the natural
  distribution of Naver Shopping ratings.** Real e-commerce ratings are
  strongly 5-star-skewed. A classifier trained on balanced priors is
  miscalibrated for deployment on the true distribution: on live traffic it
  would over-predict low ratings. Fixing this needs prior correction or
  recalibration against a representative sample, neither of which this project
  does.
- Rating 4 (9.4% of the data) has 13.5% recall under the shipped CE+argmax
  model — the model rarely predicts it. This is a real weakness of the
  published model, not smoothed over by the aggregate MAE/exact-accuracy
  numbers.
- CORAL, this project's attempt to exploit the ratings' ordinal structure,
  performs worse than plain 4-class classification here — see above for the
  likely mechanism. Ordinal regression is not a free win.
- Text alone may not contain enough information to recover a user's exact
  numeric rating.
- Bootstrap CIs quantify sampling uncertainty on this split, not uncertainty
  across different splits or retraining runs.

Project code is MIT-licensed. Dataset licensing is separate: the upstream
`bab2min/corpus` repository declares the corpus Public Domain. Users should
review the upstream source for their own use case.
