"""Central configuration. Every path is relative to the project root."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "processed" / "pharmaly_products_rag_ready.csv"
DB_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "pharmaly_products"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

TOP_K = 5
# Cosine similarity below which a result is considered unrelated.
# Calibrate it with `python calibrate.py` and override via SIMILARITY_THRESHOLD.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

MAX_HISTORY_TURNS = 3  # previous exchanges kept for follow-up questions
MIN_DESCRIPTION_CHARS = 40  # products with a shorter description are not indexed
