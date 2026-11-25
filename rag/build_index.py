"""Build the vector index ONCE from the scraped data.

Usage:
    python build_index.py            # builds the index if it does not exist
    python build_index.py --rebuild  # deletes and rebuilds it
"""
import argparse

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from . import config

def clean(value, default="Not specified"):
    if pd.isna(value) or str(value).strip() == "":
        return default
    return str(value).strip()


def load_products() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PATH)
    before = len(df)
    df = df.dropna(subset=["product_name", "description", "product_url"])
    df = df[df["description"].str.len() >= config.MIN_DESCRIPTION_CHARS]
    # Same product scraped twice (same URL) -> keep one copy.
    df = df.drop_duplicates(subset="product_url").reset_index(drop=True)
    print(f"{before} rows read, {len(df)} products kept after cleaning.")
    return df


def make_document(row) -> str:
    """One product = one chunk (descriptions average ~530 characters)."""
    return (
        f"Product: {clean(row['product_name'])}\n"
        f"Brand: {clean(row['brand'])}\n"
        f"Category: {clean(row['categories'])}\n"
        f"Current price (DA): {clean(row['current_price'])}\n"
        f"Description: {clean(row['description'])}"
    )


def main(rebuild: bool = False) -> None:
    client = chromadb.PersistentClient(path=str(config.DB_PATH))

    existing = [c.name for c in client.list_collections()]
    if config.COLLECTION_NAME in existing:
        if not rebuild:
            n = client.get_collection(config.COLLECTION_NAME).count()
            print(f"Index already exists ({n} products). Use --rebuild to redo it.")
            return
        client.delete_collection(config.COLLECTION_NAME)

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    df = load_products()
    documents = [make_document(r) for _, r in df.iterrows()]
    metadatas = [
        {
            "product_name": clean(r["product_name"], "Unknown"),
            "brand": clean(r["brand"]),
            "category": clean(r["categories"]),
            "current_price": clean(r["current_price"]),
            "product_url": clean(r["product_url"], ""),
        }
        for _, r in df.iterrows()
    ]
    ids = [f"product_{i}" for i in range(len(df))]

    print(f"Loading embedding model {config.EMBEDDING_MODEL} ...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    batch = 256
    for start in range(0, len(documents), batch):
        end = start + batch
        embeddings = model.encode(
            documents[start:end], normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings,
        )
        print(f"  indexed {min(end, len(documents))}/{len(documents)}")

    print(f"Done. {collection.count()} products stored in {config.DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    main(parser.parse_args().rebuild)
