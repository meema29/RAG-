# Pharmaly Q&A Chatbot (RAG)

A chatbot that answers questions about the products of the Pharmaly online store
(pharmaly-dz.com) using **only** the data collected by our scraper, and cites the
product pages it used. If the answer is not in the data, it says so.

## How it works

1. **Data**: `data/raw/pharmaly_products_full.csv` is the scraper output;
   `data/processed/pharmaly_products_rag_ready.csv` is the cleaned version used here.
2. **Indexing** (`build_index.py`, run once): each product (name, brand, category, price,
   description) becomes one document, embedded with the local multilingual model
   `paraphrase-multilingual-MiniLM-L12-v2` and stored in a persistent ChromaDB
   (`chroma_db/`, cosine similarity). Rows with a very short description or a
   duplicated URL are skipped.
3. **Retrieval** (`rag.py`): the question is embedded, the top 5 products are fetched,
   and those below the similarity threshold are discarded. If none remain, the bot
   refuses without calling the LLM.
4. **Generation**: `gemini-3.5-flash-lite` (via `google-genai`) answers using only the
   retrieved products, and returns which ones it relied on. If it judges the context
   insufficient it answers "not found".
5. **Sources**: product name, URL and similarity score are taken from the retrieved
   documents (never from the model's text) and printed under each answer.
6. **Follow-ups**: the last 3 exchanges are kept; a follow-up like "and its price?" is
   rewritten into a standalone question before retrieval.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then edit .env and paste your GEMINI_API_KEY
```

Get a free key at <https://aistudio.google.com>. The key is read from `.env`, which is
git-ignored; never commit it.

## Run

```bash
python build_index.py   # once (creates chroma_db/; use --rebuild to redo it)
python chatbot.py       # start chatting, type "exit" to quit
```

Optional: `python calibrate.py` prints the best similarity of in-scope and
out-of-scope questions so you can tune `SIMILARITY_THRESHOLD` in `.env`.

## Project layout

```
config.py        paths, model names, threshold
build_index.py   one-time indexing
rag.py           retrieval + generation + sources
chatbot.py       terminal interface
calibrate.py     threshold calibration helper
data/            raw and processed scraper data
demo/            demo questions and answers
rag/             exploration notebooks
```

## Demo

See `demo/qsts.md` (5+ answered questions with sources, 2+ correct refusals).

## Limits

- The Gemini free tier is rate limited; requests are retried with backoff.
- Answers are only as good as the scraped descriptions.
