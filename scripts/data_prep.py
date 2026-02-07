import os
import re
import pickle
import unicodedata
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# paths (relative to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE_DIR, "data", "np20ng.csv")
CHUNKS_DIR = os.path.join(BASE_DIR, "data", "chunks")
CHUNKS_PKL = os.path.join(CHUNKS_DIR, "nepali_news_chunks.pkl")


# validation
def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    initial_len = len(df)

    # drop rows where the main content is empty or whitespace-only
    df = df.dropna(subset=["content"])
    df = df[df["content"].str.strip().ne("")]

    # drop exact duplicates on (heading + content)
    df = df.drop_duplicates(subset=["heading", "content"])

    # flag rows where content is mostly non-Devanagari
    def devanagari_ratio(text: str) -> float:
        devs = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
        return devs / max(len(text), 1)

    df["dev_ratio"] = df["content"].apply(devanagari_ratio)
    low_quality = df[df["dev_ratio"] < 0.3]

    if len(low_quality) > 0:
        print(
            f"{len(low_quality)} rows have <30% Devanagari characters — possible encoding issues."
        )
        print(low_quality[["source", "heading", "dev_ratio"]].head(10))

    df = df.drop(columns=["dev_ratio"])

    print(
        f"Validation: {initial_len} → {len(df)} rows ({initial_len - len(df)} removed)"
    )
    return df


# cleaning
def clean_and_format(entry: dict) -> str:
    # combine header and body
    header = f"Category: {entry.get('category', 'General')} | Title: {entry.get('heading', '')}"
    body = entry.get("content", "")
    full_text = f"{header}\n{body}"

    # unicode normalization
    full_text = unicodedata.normalize("NFKC", full_text)

    # fix known encoding artifact
    full_text = full_text.replace("¥", "र्")

    # collapse whitespace
    full_text = re.sub(r"\s+", " ", full_text).strip()

    return full_text


# chunking
def build_chunks(df: pd.DataFrame) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ।", "।", "|", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
    )

    all_chunks = []
    for row in df.itertuples(index=False):
        entry = row._asdict()
        cleaned_text = clean_and_format(entry)

        metadata = {
            "source": entry.get("source", "Unknown"),
            "category": entry.get("category", "General"),
            "heading": entry.get("heading", ""),
        }

        doc_chunks = text_splitter.create_documents(
            [cleaned_text], metadatas=[metadata]
        )
        all_chunks.extend(doc_chunks)

    print(f"Total Source Docs: {len(df)}")
    print(f"Total Chunks Generated: {len(all_chunks)}")
    print(f"Average Chunks per Doc: {len(all_chunks) / len(df):.2f}")
    return all_chunks


# persistence
def save_chunks(chunks: list, path: str = CHUNKS_PKL) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {path}")


def load_chunks(path: str = CHUNKS_PKL) -> list:
    with open(path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


# CLI entry-point
def main() -> None:
    print("Loading raw data …")
    data = pd.read_csv(RAW_CSV)

    data = validate_dataframe(data)
    chunks = build_chunks(data)
    save_chunks(chunks)


if __name__ == "__main__":
    main()
