"""Gradio demo for the Korean Naver Shopping rating predictor.

Loads the published model from the Hugging Face Hub (not a local path), so
this file works unmodified once uploaded as a Space. Exposes both decoders
compared in evaluate_model.py: argmax (the classifier's native rule) and
expected value (minimizes MAE, snapped to the nearest observed rating).
"""

from __future__ import annotations

import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cringepnh/koelectra-korean-shopping-rating"
VALID_RATINGS = (1, 2, 4, 5)
RATING_ARRAY = np.array(VALID_RATINGS, dtype=np.int64)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

EXAMPLES = [
    ["배송도 빠르고 제품도 정말 좋아요", "argmax"],
    ["별로예요. 다시는 안 살 것 같습니다.", "argmax"],
    ["그냥 그래요. 나쁘지도 좋지도 않아요.", "expected value (nearest valid rating)"],
]

NO_THREE_STAR_NOTICE = (
    "**Note:** the source corpus (Naver Shopping reviews) has no 3-star "
    "reviews, so this model was trained on and can only output ratings "
    "1, 2, 4, or 5 — never 3."
)


def decode_expected_value_rounded(probabilities: np.ndarray) -> int:
    expected = float(probabilities @ RATING_ARRAY.astype(np.float64))
    distances = np.abs(expected - RATING_ARRAY.astype(np.float64))
    return int(RATING_ARRAY[np.argmin(distances)])


def predict(text: str, decoder: str) -> dict[str, float]:
    if not text or not text.strip():
        return {}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1).squeeze(0).numpy()

    labels = {str(int(model.config.id2label[i])): float(p) for i, p in enumerate(probabilities)}

    if decoder == "expected value (nearest valid rating)":
        predicted = decode_expected_value_rounded(probabilities)
        # Highlight the decoded rating rather than replacing the distribution,
        # since expected-value decoding doesn't produce its own probabilities.
        # ASCII-only label suffix: a non-ASCII arrow here previously crashed
        # any caller printing this dict through a non-UTF-8 console/log
        # encoding (e.g. Windows cp1251), which is a real failure mode for a
        # deployed demo, not just a display nicety.
        return {
            f"{k} (predicted)" if k == str(predicted) else k: v for k, v in labels.items()
        }
    return labels


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Korean shopping review", lines=3, placeholder="이 제품 어땠나요?"),
        gr.Radio(
            choices=["argmax", "expected value (nearest valid rating)"],
            value="argmax",
            label="Decoder (see decoding.py / README for why these differ)",
        ),
    ],
    outputs=gr.Label(label="Predicted rating probabilities", num_top_classes=4),
    examples=EXAMPLES,
    title="Korean Shopping Review Rating Predictor",
    description=(
        "4-class rating prediction (1, 2, 4, or 5 stars) on real Naver "
        "Shopping reviews, fine-tuned from the raw KoELECTRA v3 "
        "discriminator. MAE 0.386, exact accuracy 71.7% on a held-out test "
        "set of 19,982 reviews (argmax decoder). "
        + NO_THREE_STAR_NOTICE
        + " See the [model card](https://huggingface.co/cringepnh/koelectra-korean-shopping-rating) "
        "and [source](https://github.com/cringepnh/korean-shopping-review-rating-predictor) "
        "for the full evaluation, including the CORAL ordinal-regression comparison."
    ),
)

if __name__ == "__main__":
    demo.launch()
