---
language:
- ko
license: mit
library_name: transformers
pipeline_tag: text-classification
tags:
- korean
- shopping-reviews
- rating-classification
- koelectra
metrics:
- mae
- rmse
- accuracy
base_model: monologg/koelectra-base-v3-discriminator
---

# Korean Shopping Review Rating Classifier

Four-class classifier for real Korean Naver Shopping ratings **1, 2, 4, and
5**. This is not a movie model. Rating 3 is absent
from the source dataset, so the model cannot predict it.

Source code, full reproduction pipeline, and a from-scratch CORAL
ordinal-regression comparison (which this model outperforms — see below):
[GitHub](https://github.com/cringepnh/korean-shopping-review-rating-predictor).
A Gradio demo lives in that repo's `space/` folder and loads this model
straight from the Hub, but **there is no hosted demo** — Hugging Face requires
a PRO subscription to run Gradio Spaces, so the demo is local-only.

## Results

All rows use the same deterministic 19,982-review test split, with 95%
bootstrap confidence intervals. This model is the **CE + argmax** row —
loadable with stock `AutoModelForSequenceClassification`, no custom code.

| System | MAE | Exact accuracy | QWK |
|---|---:|---:|---:|
| Constant: training median | 1.5859 | 9.39% | 0.000 |
| Majority rating class | 1.7735 | 40.62% | 0.000 |
| **This model (CE + argmax)** | **0.3856** [0.376, 0.396] | **71.66%** [71.0, 72.3] | 0.837 |
| CE + median decoder (same model, different decode rule) | 0.3858 | 70.88% | **0.842** |
| CORAL ordinal regression (separate model, not published here) | 0.5632 | 56.89% | 0.810 |

The best checkpoint is epoch 2, selected on validation MAE (0.3894). Full
per-class breakdown, confusion matrices, the CORAL comparison, and the
decoder analysis (argmax vs. expected value vs. median — an earlier version
of this project incorrectly assumed expected value minimizes MAE; it
minimizes RMSE instead) are in the
[README](https://github.com/cringepnh/korean-shopping-review-rating-predictor#method-and-evaluation-results).

**Known weakness:** rating 4 (9.4% of the data) has only 13.5% recall — the
model rarely predicts it. This is not hidden by the aggregate MAE.

## Data and provenance

Source: [`bab2min/corpus` Naver Shopping sentiment
file](https://github.com/bab2min/corpus/blob/master/sentiment/naver_shopping.txt).
The upstream repository declares the corpus Public Domain.

Pinned source SHA-256:
`dc4d1aca0a148671cbe80bcb81962eee297370acab42be93c1617ce9336479c0`

The source contains 200,000 real source-file ratings. After excluding all 184
rows associated with 92 identical texts that have conflicting ratings, the
splits are 159,852 train / 19,982 validation / 19,982 test. Targets are never
derived from sentiment labels or generated synthetically.

## Use

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "cringepnh/koelectra-korean-shopping-rating"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
inputs = tokenizer("배송도 빠르고 제품도 정말 좋아요", return_tensors="pt", truncation=True)
with torch.inference_mode():
    probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
index = int(probabilities.argmax())
print(model.config.id2label[index], float(probabilities[index]))
```

## Limitations

- Naver Shopping domain only; not movie reviews.
- No 3-star examples or output class.
- **Curated class priors.** The upstream corpus dropped 3-star reviews as
  ambiguous and sampled positive (4–5) against negative (1–2) to roughly 1:1.
  Real Naver Shopping ratings are strongly 5-star-skewed, so this model is
  miscalibrated for the natural distribution and will over-predict low ratings
  on live traffic. Apply prior correction before deploying.
- Rating 4 has 13.5% recall under this model's argmax decoder — a real
  weakness, not smoothed over by MAE.
- This uses ordinary 4-class cross-entropy with an argmax decoder by default.
  A from-scratch CORAL ordinal-regression model was trained as a comparison
  and performed *worse* (MAE 0.5632 vs. 0.3856); see the GitHub README for
  the full result and the likely mechanism (CORAL's shared projection
  collapses per-class recall toward the two extreme ratings).
- Code is MIT; dataset licensing is the upstream repository's separate Public
  Domain declaration.
