"""
RAG chain: retrieves relevant GDPR chunks from FAISS, then answers
user questions in English or Chinese with article citations.
"""

import os

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

load_dotenv()

FAISS_DIR = "faiss_index"
TOP_K = 6

SYSTEM_PROMPT = """\
You are a GDPR legal expert assistant. Answer the user's question using ONLY the GDPR article text provided below.

Rules:
1. Detect the language of the question and respond in the SAME language.
   - If the question is in Chinese (中文), respond entirely in Chinese.
   - If the question is in English, respond in English.
2. Ground every claim in the provided GDPR context. Do not invent or extrapolate.
3. If the context does not contain enough information, say so honestly.
4. At the end of your answer, include a "📌 Sources / 引用来源" section listing each cited article:
   - Article number, title, and URL.

GDPR Context:
{context}
"""

HUMAN_PROMPT = "{question}"


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        m = doc.metadata
        parts.append(
            f"[Article {m['article_number']}: {m['title']}]\n"
            f"URL: {m['source']}\n\n"
            f"{doc.page_content}"
        )
    return "\n\n" + ("─" * 60 + "\n\n").join(parts)


def load_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        FAISS_DIR, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def build_chain():
    retriever = load_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever
