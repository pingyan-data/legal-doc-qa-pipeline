# GDPR Q&A Pipeline

A Retrieval-Augmented Generation (RAG) system that lets you ask questions about the General Data Protection Regulation in **English or Chinese**, with answers grounded in the actual GDPR article text and full source citations.

Live demo: [Streamlit Cloud](#) <!-- update after deploy -->

---

## Architecture

```
+-------------------------------------------------------------+
|                      OFFLINE (run once)                     |
|                                                             |
|  gdpr-info.eu --> scraper.py --> gdpr_articles.json         |
|                                        |                    |
|                                   ingest.py                 |
|                                        |                    |
|                            OpenAI Embeddings                |
|                         (text-embedding-3-small)            |
|                                        |                    |
|                                   FAISS index               |
|                                  (faiss_index/)             |
+---------------------------------------+---------------------+
                                        |
+---------------------------------------v---------------------+
|                       ONLINE (app.py)                       |
|                                                             |
|  User Query --> OpenAI Embeddings --> FAISS Retriever       |
|      (EN/ZH)                               |               |
|                                     Top-6 Chunks           |
|                                            |               |
|                              LangChain LCEL Chain          |
|                                            |               |
|                          GPT-4o-mini + System Prompt       |
|                          (detects language, cites sources) |
|                                            |               |
|                              Streamed Answer + Sources     |
|                                            |               |
|                              Streamlit UI (app.py)         |
+-------------------------------------------------------------+
```

### Components

| File | Role |
|------|------|
| `scraper.py` | Scrapes all 99 GDPR articles from gdpr-info.eu |
| `ingest.py` | Chunks text, embeds with OpenAI, persists to FAISS |
| `chain.py` | LangChain LCEL RAG chain with bilingual system prompt |
| `app.py` | Streamlit UI with streaming answers and source cards |

---

## Quick Start (local)

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-...
```

### 3. Scrape GDPR articles (run once, ~2 min)

```bash
python scraper.py
# Produces: gdpr_articles.json (99 articles)
```

### 4. Build the vector store (run once, ~1 min)

```bash
python ingest.py
# Produces: faiss_index/ directory
```

### 5. Launch the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Deploy to Streamlit Cloud

The `faiss_index/` is committed to this repo, so no scraping or ingestion is needed on the cloud.

1. Fork or push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo
3. In the app settings, add a secret:

```toml
OPENAI_API_KEY = "sk-..."
```

4. Deploy — the app will be live in about 1 minute.

---

## Example Queries

| Language | Query |
|----------|-------|
| English | What are the rights of data subjects under GDPR? |
| English | What is the right to be forgotten? |
| English | What are the maximum fines for GDPR violations? |
| Chinese | 个人数据泄露后，数据控制者有多少小时必须向监管机构报告？ |
| Chinese | GDPR 对数据保护官（DPO）有什么要求？ |
| Chinese | 什么是数据可携带权？ |

---

## Cost Estimate

| Step | Model | Approx. Cost |
|------|-------|--------------|
| Ingestion (one-time) | text-embedding-3-small | ~$0.01 |
| Per query | gpt-4o-mini | ~$0.001 |

---

## Project Structure

```
legal-doc-qa-pipeline/
├── scraper.py          # Web scraper for gdpr-info.eu
├── ingest.py           # Chunking + embedding + FAISS ingestion
├── chain.py            # LangChain LCEL RAG chain
├── app.py              # Streamlit frontend
├── faiss_index/        # Pre-built vector index (committed)
├── requirements.txt
├── .env.example
└── .streamlit/
    └── secrets.toml.example
```
