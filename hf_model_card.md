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

## Results

All rows use the same deterministic 19,982-review test split.

| System | Prediction | MAE | RMSE | Exact accuracy | Within ±1 |
|---|---:|---:|---:|---:|---:|
| Constant: training median | 4.0000 | 1.5859 | 1.8182 | 9.39% | 50.02% |
| Constant: training mean | 3.2265 | 1.5862 | 1.6454 | 0.00% | 9.39% |
| Majority rating class | 5 | 1.7735 | 2.4192 | 40.62% | 50.02% |
| Fine-tuned 4-class KoELECTRA | — | **0.3856** | **0.8364** | **71.66%** | **94.46%** |

The best checkpoint is epoch 2, selected on validation MAE (0.3894). Epoch 3
was slightly worse. Test was evaluated only after checkpoint selection.

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

model_id = "PATH_OR_HUB_ID"
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
- Rating classes are imbalanced.
- This uses ordinary 4-class cross-entropy; it has no order-aware loss.
- Code is MIT; dataset licensing is the upstream repository's separate Public
  Domain declaration.
