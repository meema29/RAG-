"""Retrieval + generation. No side effects at import time."""
import json
import os
import time
from functools import lru_cache

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from sentence_transformers import SentenceTransformer

from . import config
load_dotenv(config.ROOT / ".env")

NOT_FOUND_MESSAGE = (
    "Sorry, I couldn't find this information in the Pharmaly product database."
)

SYSTEM_INSTRUCTION = """You are Pharmaly, an assistant that answers questions about the
products of the Pharmaly online pharmacy/parapharmacy, using ONLY the numbered
PRODUCTS given in the user message (they come from the Pharmaly website).

Rules:
1. Use only the information in the PRODUCTS. Never use general knowledge.
2. Never invent product names, ingredients, prices, brands or properties.
3. If the PRODUCTS do not contain the answer, set "answer" to exactly:
   "NOT_IN_DATA" and "used" to [].
4. If the information is incomplete, say what is missing.
5. When the user asks for multiple products, a list, or uses plural
   wording (e.g. "which products"), include ALL relevant products
   found in the provided PRODUCTS, not just one. Do not omit
   relevant products from the retrieved context.
6. LANGUAGE RULE (IMPORTANT):
   - Identify the language of the user's latest question.
   - Write the entire answer in that same language.
   - If the question is in English, answer entirely in English.
   - If the question is in French, answer entirely in French.
   - If the question is in Arabic, answer entirely in Arabic.
   - Do not switch to French by default.
   - Product names and brand names must remain unchanged.
   - This rule applies even if previous conversation messages
     were written in another language.
7. Prices are in Algerian dinars (DA).
8. In "used", list the numbers of the PRODUCTS you actually relied on.
Return JSON: {"answer": string, "used": [int, ...]}"""


# ---------- lazy singletons (loaded once per process) ----------
@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(config.DB_PATH))
    try:
        return client.get_collection(config.COLLECTION_NAME)
    except Exception as exc:  # collection missing
        raise RuntimeError(
            "Vector index not found. Run `python build_index.py` first."
        ) from exc


@lru_cache(maxsize=1)
def _llm() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return genai.Client(api_key=key)


def _generate(contents: str, system_instruction: str | None = None, json_mode=False,
              retries: int = 4) -> str:
    """Call Gemini, retrying with exponential backoff on rate limits / overload."""
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        response_mime_type="application/json" if json_mode else None,
    )
    delay = 5
    for attempt in range(retries + 1):
        try:
            resp = _llm().models.generate_content(
                model=config.LLM_MODEL, contents=contents, config=cfg
            )
            return resp.text or ""
        except errors.APIError as exc:
            retryable = getattr(exc, "code", None) in (429, 500, 503)
            if not retryable or attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2
    return ""


# ---------- retrieval ----------
def retrieve(query: str, top_k: int = config.TOP_K,
             threshold: float = config.SIMILARITY_THRESHOLD) -> dict:
    query = query.strip()
    if not query:
        raise ValueError("Empty question.")
    vec = _embedder().encode([query], normalize_embeddings=True).tolist()
    res = _collection().query(
        query_embeddings=vec, n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = [
        {"content": d, "metadata": m, "similarity": 1 - dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    kept = [h for h in hits if h["similarity"] >= threshold]
    return {
        "hits": kept,
        "max_similarity": max((h["similarity"] for h in hits), default=None),
    }


# ---------- conversation handling ----------
def _standalone_question(question: str, history: list[tuple[str, str]]) -> str:
    """Rewrite a follow-up ("and its price?") into a self-contained question."""
    if not history:
        return question
    convo = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-config.MAX_HISTORY_TURNS:])
    prompt = (
        "Rewrite the last user question so it can be understood without the "
        "conversation. Keep its language. If it is already standalone, return it "
        f"unchanged. Output only the question.\n\n{convo}\n\nLast question: {question}"
    )
    rewritten = _generate(prompt).strip()
    return rewritten or question


# ---------- main entry point ----------
def ask(question: str, history: list[tuple[str, str]] | None = None) -> dict:
    """Return {"answer", "sources": [{name, url, similarity}], "max_similarity"}."""
    history = history or []
    standalone = _standalone_question(question, history)
    result = retrieve(standalone)

    if not result["hits"]:
        return {"answer": NOT_FOUND_MESSAGE, "sources": [],
                "max_similarity": result["max_similarity"], "query": standalone}

    context = "\n\n".join(
        f"PRODUCT {i}\n{h['content']}\nURL: {h['metadata'].get('product_url', '')}"
        for i, h in enumerate(result["hits"], start=1)
    )
    raw = _generate(
        f"PRODUCTS:\n{context}\n\nQUESTION: {standalone}",
        system_instruction=SYSTEM_INSTRUCTION, json_mode=True,
    )
    try:
        data = json.loads(raw)
        answer = str(data.get("answer", "")).strip()
        used = [int(i) for i in data.get("used", [])]
    except (json.JSONDecodeError, ValueError, TypeError):
        answer, used = raw.strip(), list(range(1, len(result["hits"]) + 1))

    if answer == "NOT_IN_DATA" or not answer:
        return {"answer": NOT_FOUND_MESSAGE, "sources": [],
                "max_similarity": result["max_similarity"], "query": standalone}

    # Sources come from the retrieved documents, never from the model's text.
    sources = [
        {"name": h["metadata"]["product_name"], "url": h["metadata"]["product_url"],
         "similarity": h["similarity"]}
        for i, h in enumerate(result["hits"], start=1) if i in used
    ] or [
        {"name": h["metadata"]["product_name"], "url": h["metadata"]["product_url"],
         "similarity": h["similarity"]}
        for h in result["hits"]
    ]
    return {"answer": answer, "sources": sources,
            "max_similarity": result["max_similarity"], "query": standalone}
