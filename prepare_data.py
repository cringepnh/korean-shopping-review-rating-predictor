"""Create deterministic, disjoint train/validation/test rating splits."""

from __future__ import annotations

import json

from sklearn.model_selection import train_test_split

from data_utils import (
    SEED,
    SOURCE_SHA256,
    SPLIT_MANIFEST_PATH,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    load_real_reviews,
    print_target_distribution,
    sets_are_disjoint,
)


def main() -> None:
    source = load_real_reviews()
    train, temporary = train_test_split(
        source,
        test_size=0.20,
        random_state=SEED,
        stratify=source["rating"],
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=SEED,
        stratify=temporary["rating"],
    )
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_text = set(train["review"])
    validation_text = set(validation["review"])
    test_text = set(test["review"])
    if not sets_are_disjoint(train_text, validation_text, test_text):
        raise AssertionError("Prepared splits contain text leakage")

    for name, frame in (("TRAIN", train), ("VALIDATION", validation), ("TEST", test)):
        print_target_distribution(name, frame)
    train.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    validation.to_csv(VALIDATION_PATH, index=False, encoding="utf-8")
    test.to_csv(TEST_PATH, index=False, encoding="utf-8")

    manifest = {
        "source_sha256": SOURCE_SHA256,
        "seed": SEED,
        "split": "80/10/10 stratified after text deduplication",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "target_origin": "source_file",
        "text_disjoint": True,
    }
    SPLIT_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved deterministic splits and {SPLIT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
