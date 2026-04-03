from __future__ import annotations

import pickle
import re
import unicodedata

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    initial_len = len(df)

    df = df.dropna(subset=["content"])
    df = df[df["content"].str.strip().ne("")]
    df = df.drop_duplicates(subset=["heading", "content"])

    def devanagari_ratio(text: str) -> float:
        devs = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
        return devs / max(len(text), 1)

    df["dev_ratio"] = df["content"].apply(devanagari_ratio)
    low_quality = df[df["dev_ratio"] < 0.3]

    if len(low_quality) > 0:
        print(
            f"{len(low_quality)} rows have <30% Devanagari characters - possible encoding issues."
        )
        print(low_quality[["source", "heading", "dev_ratio"]].head(10))

    df = df.drop(columns=["dev_ratio"])

    print(
        f"Validation: {initial_len} -> {len(df)} rows ({initial_len - len(df)} removed)"
    )
    return df


def clean_and_format(entry: dict) -> str:
    header = (
        f"Category: {entry.get('category', 'General')} | "
        f"Title: {entry.get('heading', '')}"
    )
    body = entry.get("content", "")
    full_text = f"{header}\n{body}"

    full_text = unicodedata.normalize("NFKC", full_text)
    full_text = full_text.replace("¥", "र्")
    full_text = re.sub(r"\s+", " ", full_text).strip()
    return full_text


def build_chunks(df: pd.DataFrame) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ।", "।", "|", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
    )

    all_chunks: list[Document] = []
    for row in df.itertuples(index=False):
        entry = row._asdict()
        cleaned_text = clean_and_format(entry)

        metadata = {
            "source": entry.get("source", "Unknown"),
            "category": entry.get("category", "General"),
            "heading": entry.get("heading", ""),
        }

        doc_chunks = text_splitter.create_documents([cleaned_text], metadatas=[metadata])
        all_chunks.extend(doc_chunks)

    print(f"Total Source Docs: {len(df)}")
    print(f"Total Chunks Generated: {len(all_chunks)}")
    print(f"Average Chunks per Doc: {len(all_chunks) / len(df):.2f}")
    return all_chunks


def save_chunks(chunks: list[Document], path: str) -> None:
    with open(path, "wb") as file:
        pickle.dump(chunks, file)
    print(f"Saved {len(chunks)} chunks to {path}")


def load_chunks(path: str) -> list[Document]:
    with open(path, "rb") as file:
        chunks = pickle.load(file)
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks
