"""
Load scraped GDPR articles, chunk, embed, and persist to ChromaDB.
Run after scraper.py: python ingest.py
"""

import json
import os

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

ARTICLES_FILE = "gdpr_articles.json"
FAISS_DIR = "faiss_index"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 80


def load_documents(path: str = ARTICLES_FILE) -> list[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scraper.py` first."
        )
    with open(path, encoding="utf-8") as f:
        articles = json.load(f)

    docs = []
    for article in articles:
        doc = Document(
            page_content=article["text"],
            metadata={
                "article_number": article["article_number"],
                "title": article["title"],
                "source": article["url"],
            },
        )
        docs.append(doc)
    print(f"Loaded {len(docs)} articles.")
    return docs


def build_vectorstore(docs: list[Document]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_DIR)
    print(f"Persisted vector store to {FAISS_DIR}/")
    return vectorstore


if __name__ == "__main__":
    docs = load_documents()
    build_vectorstore(docs)
    print("Ingestion complete. You can now run the app: streamlit run app.py")
