from __future__ import annotations

from pathlib import Path

import pandas as pd

from nepali_news_rag.config import get_settings
from nepali_news_rag.data_prep import build_chunks, save_chunks, validate_dataframe


def main() -> None:
    settings = get_settings()

    if not settings.raw_csv_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {settings.raw_csv_path}. Place CSV in data/raw first."
        )

    settings.chunks_pkl_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw CSV: {settings.raw_csv_path}")
    data = pd.read_csv(settings.raw_csv_path)
    data = validate_dataframe(data)
    chunks = build_chunks(data)
    save_chunks(chunks, str(settings.chunks_pkl_path))
    print("Data preparation completed.")


if __name__ == "__main__":
    main()
