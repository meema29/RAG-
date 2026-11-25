# Pharmaly Q&A Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about products from the Pharmaly online store ([pharmaly-dz.com](https://pharmaly-dz.com)) using data collected by our scraper.

The chatbot retrieves relevant products, generates answers using Gemini, and provides links to the product pages used as sources. If the requested information is unavailable, it communicates that instead of inventing an answer.

## How it works

1. **Data preparation:** The scraper output is stored in `data/raw/pharmaly_products_full.csv`. The cleaned dataset used by the chatbot is `data/processed/pharmaly_products_rag_ready.csv`.
2. **Indexing:** Each product is converted into a document containing its name, brand, category, price, and description. Documents are embedded using `paraphrase-multilingual-MiniLM-L12-v2` and stored in a persistent ChromaDB collection using cosine similarity.
3. **Retrieval:** The user's question is embedded, and the five most relevant products are retrieved. Products below the configured similarity threshold are discarded.
4. **Generation:** Gemini generates an answer using only the retrieved product information. The chatbot includes relevant product sources and refuses questions that are outside the available data.
5. **Follow-up questions:** The chatbot maintains the last three conversation exchanges and rewrites follow-up questions into standalone questions before retrieval.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/meema29/RAG-.git
cd RAG-
```

### 2. Create and activate a virtual environment

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

On Linux / macOS:

```bash
cp .env.example .env
```

Open `.env` and add your Gemini API key.

You can obtain a key from [Google AI Studio](https://aistudio.google.com).

**Never commit your `.env` file or share your API key.**

## Running the project

### 1. Build the vector index

The ChromaDB database is excluded from Git because it is generated locally.

From the project root, run:

Windows:

```powershell
py -3.12 -m rag.build_index
```

Linux / macOS:

```bash
python -m rag.build_index
```

The script creates the persistent vector index from the processed product dataset.

If an index already exists, the script detects it and skips indexing. To rebuild it intentionally, use:

```powershell
py -3.12 -m rag.build_index --rebuild
```

### 2. Start the chatbot

Open `rag/chatbot.ipynb` in Jupyter Notebook or VS Code with the Jupyter extension.

Run the notebook cells in order and start asking questions about Pharmaly products.

Type `exit` to end the conversation.

## Optional: Similarity threshold calibration

To compare similarity scores for relevant and irrelevant questions, run:

Windows:

```powershell
py -3.12 -m rag.calibrate
```

Linux / macOS:

```bash
python -m rag.calibrate
```

The script displays the highest similarity scores for out-of-scope questions and the lowest scores for in-scope questions. These results can help tune `SIMILARITY_THRESHOLD` in `.env`.

## Project structure

```text
RAG-/
│
├── data/
│   ├── raw/
│   │   └── pharmaly_products_full.csv
│   └── processed/
│       └── pharmaly_products_rag_ready.csv
│
├── demo/
│   └── qsts.md
│
├── rag/
│   ├── chroma_db/          Generated vector database (not committed)
│   ├── build_index.py      Index construction
│   ├── calibrate.py        Similarity threshold calibration
│   ├── chatbot.ipynb       Interactive chatbot interface
│   ├── config.py           Central configuration
│   ├── rag.py              Retrieval and generation pipeline
│   └── *.ipynb             Development and experimentation notebooks
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Demo

See `demo/qsts.md` for example questions, answers, product sources, and out-of-scope refusals.

## Limitations

- Answers depend on the completeness and quality of the scraped product descriptions.
- Retrieval quality depends on the embedding model and similarity threshold.
- Gemini API usage is subject to model availability and rate limits.