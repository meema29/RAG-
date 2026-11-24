
import chromadb
from embeddings import get_embedding


# -----------------------------------------
# 1. Configuration
# -----------------------------------------

DB_PATH = "./chroma_db"
COLLECTION_NAME = "pharmaly_products"

DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.50


# -----------------------------------------
# 2. Connect to ChromaDB
# -----------------------------------------

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# -----------------------------------------
# 3. Process the user query
# -----------------------------------------

def preprocess_query(query):
    """
    Validate and clean the user's question.
    """

    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    return query


# -----------------------------------------
# 4. Retrieve relevant documents
# -----------------------------------------

def retrieve_documents(query_embedding, top_k=DEFAULT_TOP_K):
    """
    Retrieve the top-k most relevant documents
    from ChromaDB.
    """

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_documents = []

    for i in range(len(results["ids"][0])):

        distance = results["distances"][0][i]

        document = {
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": distance,
            "similarity": 1 - distance
        }

        retrieved_documents.append(document)

    return retrieved_documents


# -----------------------------------------
# 5. Apply the retrieval threshold
# -----------------------------------------

def apply_retrieval_threshold(
    retrieved_documents,
    threshold=DEFAULT_SIMILARITY_THRESHOLD
):
    """
    Filter documents according to their similarity scores.
    """

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if not retrieved_documents:
        return {
            "status": "insufficient_evidence",
            "documents": [],
            "max_similarity": None,
            "threshold": threshold
        }

    accepted_documents = [
        document
        for document in retrieved_documents
        if document["similarity"] >= threshold
    ]

    max_similarity = max(
        document["similarity"]
        for document in retrieved_documents
    )

    status = (
        "sufficient_evidence"
        if accepted_documents
        else "insufficient_evidence"
    )

    return {
        "status": status,
        "documents": accepted_documents,
        "max_similarity": max_similarity,
        "threshold": threshold
    }


# -----------------------------------------
# 6. Complete retrieval pipeline
# -----------------------------------------

def search_products(
    query,
    top_k=DEFAULT_TOP_K,
    threshold=DEFAULT_SIMILARITY_THRESHOLD
):
    """
    Process a user question, generate its embedding,
    retrieve relevant products and apply the threshold.

    This is the main function Member 4 can call.
    """

    cleaned_query = preprocess_query(query)

    query_embedding = get_embedding(cleaned_query)

    retrieved_documents = retrieve_documents(
        query_embedding,
        top_k=top_k
    )

    threshold_result = apply_retrieval_threshold(
        retrieved_documents,
        threshold=threshold
    )

    return {
        "query": cleaned_query,
        "status": threshold_result["status"],
        "documents": threshold_result["documents"],
        "max_similarity": threshold_result["max_similarity"],
        "threshold": threshold_result["threshold"]
    }
