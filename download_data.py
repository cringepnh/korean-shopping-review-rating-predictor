"""Download the real Naver Shopping rating corpus and verify its identity."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import requests

from data_utils import (
    DATA_DIR,
    EXPECTED_SOURCE_ROWS,
    LICENSE_PAGE,
    RAW_DATA_PATH,
    SOURCE_MANIFEST_PATH,
    SOURCE_PAGE,
    SOURCE_SHA256,
    SOURCE_URL,
    VALID_RATINGS,
    load_real_reviews,
    sha256_file,
)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    temporary_path = RAW_DATA_PATH.with_suffix(".tsv.part")
    print(f"Downloading real rating data from:\n  {SOURCE_URL}")
    try:
        with requests.get(SOURCE_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    actual_hash = sha256_file(temporary_path)
    if actual_hash != SOURCE_SHA256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Source SHA-256 mismatch: expected {SOURCE_SHA256}, got {actual_hash}. "
            "No fallback data will be generated."
        )
    temporary_path.replace(RAW_DATA_PATH)

    manifest = {
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "license_declared_by_source_repository": "Public Domain",
        "license_page": LICENSE_PAGE,
        "sha256": SOURCE_SHA256,
        "expected_rows": EXPECTED_SOURCE_ROWS,
        "ratings_present": list(VALID_RATINGS),
        "target_origin": "source_file",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    SOURCE_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame = load_real_reviews()
    print(f"\nVerified {len(frame):,} unique real-rating reviews.")
    print(f"Manifest: {SOURCE_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
