---
license: mit
language:
  - ko
library_name: transformers
tags:
  - korean
  - sentiment-analysis
  - movie-reviews
  - koelectra
  - ordinal-classification
  - pytorch
datasets:
  - nsmc
base_model: monologg/koelectra-base-v3-discriminator
metrics:
  - mae
  - accuracy
pipeline_tag: text-classification
---

# Korean Movie Review Rating Predictor 🎬

Predict the score (1–10) a viewer would give a Korean movie based on their review text.

Fine-tuned from [`monologg/koelectra-base-v3-discriminator`](https://huggingface.co/monologg/koelectra-base-v3-discriminator) using ordinal classification (10-class softmax) on 153k Korean movie reviews from the [NSMC dataset](https://github.com/e9t/nsmc).

## Results

| Metric | Value |
|--------|-------|
| **MAE** | **1.29** |
| **RMSE** | **2.24** |
| **Accuracy ±1** | **79.98%** |

> Evaluated on 19,195 held-out test samples.

## Usage

```python
import torch
from transformers import ElectraModel, ElectraTokenizer
import torch.nn as nn

# 1. Define the model architecture (must match training)
class KoELECTRAOrdinalClassifier(nn.Module):
    def __init__(self, model_name, num_classes=10):
        super().__init__()
        self.electra = ElectraModel.from_pretrained(model_name)
        hidden = self.electra.config.hidden_size  # 768
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_emb)

# 2. Load model
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
model = KoELECTRAOrdinalClassifier(MODEL_NAME, num_classes=10)
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()

# 3. Tokenize and predict
tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
text = "정말 재미있는 영화였습니다! 배우들의 연기가 훌륭해요."

enc = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
with torch.no_grad():
    logits = model(enc["input_ids"], enc["attention_mask"])
    probs = torch.softmax(logits, dim=1).squeeze()
    score = int(torch.argmax(probs)) + 1  # 1-10

print(f"Predicted score: {score}/10")
print(f"Confidence: {probs[score-1]*100:.1f}%")
```

## Training Details

| Parameter | Value |
|---|---|
| Base model | `monologg/koelectra-base-v3-discriminator` |
| Training data | 153,556 reviews |
| Epochs | 5 (best at epoch 2) |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Optimizer | AdamW (weight decay 0.01) |
| Loss | CrossEntropyLoss |
| Hardware | NVIDIA RTX 4080 Laptop GPU |

## Limitations

- **No scores 5 or 6**: NSMC removed "neutral" reviews (scores 5-8) when creating the dataset. Our mapping covers 1-4 and 7-10 only.
- **Korean only**: The model was trained exclusively on Korean text.

## Links

- **GitHub**: [cringepnh/Korean-Movie-Review-Rating-Predictor](https://github.com/cringepnh/Korean-Movie-Review-Rating-Predictor)
- **Base model**: [monologg/koelectra-base-v3-discriminator](https://huggingface.co/monologg/koelectra-base-v3-discriminator)
