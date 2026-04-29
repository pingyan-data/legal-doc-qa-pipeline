"""
Streamlit UI for the GDPR Q&A RAG system.
Run: streamlit run app.py
"""

import os
import streamlit as st

# Inject Streamlit Cloud secrets into env before any OpenAI/LangChain imports
if "OPENAI_API_KEY" in st.secrets:
    os.environ.setdefault("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])

from chain import build_chain

st.set_page_config(
    page_title="GDPR Q&A | GDPR 问答系统",
    page_icon="⚖️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #4A90E2;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚖️ GDPR Q&A System")
st.caption(
    "Ask questions about the General Data Protection Regulation in **English or Chinese**. "
    "用**中文或英文**提问 GDPR 相关问题。"
)
st.divider()


@st.cache_resource(show_spinner="Loading GDPR knowledge base...")
def get_chain():
    if not os.path.exists("faiss_index"):
        return None, None
    return build_chain()


chain, retriever = get_chain()

if chain is None:
    st.error(
        "Vector database not found. Please run the pipeline first:\n\n"
        "```\npython scraper.py\npython ingest.py\n```"
    )
    st.stop()

EXAMPLES = [
    "What are the rights of data subjects under GDPR?",
    "个人数据泄露发生后，数据控制者有多少小时必须向监管机构报告？",
    "What is the legal basis for processing personal data?",
    "GDPR 对数据保护官（DPO）有什么要求？",
    "What are the maximum fines for GDPR violations?",
]

with st.expander("📋 Example questions / 示例问题", expanded=False):
    for ex in EXAMPLES:
        if st.button(ex, key=ex):
            st.session_state["query_input"] = ex

query = st.text_area(
    label="Your question / 您的问题",
    key="query_input",
    placeholder="e.g. What is the right to be forgotten? / 什么是被遗忘权？",
    height=90,
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Ask / 提问", type="primary", use_container_width=True)

if submit and query.strip():
    with st.spinner("Searching GDPR articles... / 正在检索 GDPR 条款..."):
        source_docs = retriever.invoke(query)
        answer_placeholder = st.empty()
        full_answer = ""
        for chunk in chain.stream(query):
            full_answer += chunk
            answer_placeholder.markdown(full_answer + "▌")
        answer_placeholder.markdown(full_answer)

    st.divider()
    st.markdown("### 📚 Retrieved Articles / 检索到的条款")

    seen: set[int] = set()
    for doc in source_docs:
        art_num = doc.metadata["article_number"]
        if art_num in seen:
            continue
        seen.add(art_num)
        title = doc.metadata["title"]
        url = doc.metadata["source"]
        with st.expander(f"Article {art_num}: {title}"):
            st.markdown(
                f'<div class="source-card">'
                f'<a href="{url}" target="_blank">🔗 {url}</a>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.text(doc.page_content[:420].strip() + " ...")

elif submit:
    st.warning("Please enter a question. / 请输入问题。")

st.divider()
st.caption(
    "Data sourced from [gdpr-info.eu](https://gdpr-info.eu) · "
    "Powered by OpenAI + LangChain + FAISS"
)
