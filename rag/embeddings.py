from sentence_transformers import SentenceTransformer

# Embedding model configuration
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded!")
print("Embedding dimensions:", model.get_sentence_embedding_dimension())


def get_embeddings(texts):
    """
    Convert a list of texts into embeddings.
    Returns a list of numerical vectors.
    """

    if not texts:
        return []

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    return embeddings.tolist()


def get_embedding(text):
    """
    Convert one text into one embedding.
    """

    return get_embeddings([text])[0]


if __name__ == "__main__":
    example = "Vitamin C supplement for immunity"

    vector = get_embedding(example)

    print("Embedding generated successfully!")
    print("Vector dimensions:", len(vector))
    print("First 5 values:", vector[:5])
