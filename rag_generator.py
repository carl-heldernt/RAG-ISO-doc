"""Refined for Adaptive Bilingual Output + Single Collection"""
import argparse
import os
import re

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# ------------------------------------------------------------
# Load environment
# ------------------------------------------------------------
load_dotenv()

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_openai_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

missing = [
    k
    for k, v in {
        "AZURE_OPENAI_API_KEY": azure_openai_api_key,
        "AZURE_OPENAI_ENDPOINT": azure_openai_endpoint,
        "AZURE_OPENAI_API_VERSION": azure_openai_api_version,
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": azure_openai_chat_deployment,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": azure_openai_embedding_deployment,
    }.items()
    if not v
]
if missing:
    raise ValueError(f"❌ Missing env vars: {', '.join(missing)}")

# ------------------------------------------------------------
# LLM + Embeddings
# ------------------------------------------------------------
llm = AzureChatOpenAI(
    openai_api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    azure_deployment=azure_openai_chat_deployment,
    openai_api_version=azure_openai_api_version,
)

embeddings = AzureOpenAIEmbeddings(
    openai_api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    azure_deployment=azure_openai_embedding_deployment,
    openai_api_version=azure_openai_api_version,
)

# ------------------------------------------------------------
# Prompt Templates
# ------------------------------------------------------------
bilingual_prompt = ChatPromptTemplate.from_template("""
You are an expert assistant.

Use the retrieved context below to answer the question precisely.
If the retrieved text does not contain enough information, respond:
"Insufficient information found in retrieved documents."

Provide the answer in **Bilingual format (English + Traditional Chinese)**.
Use clear structure with English first, then Traditional Chinese.

Retrieved text:
{context}

Question:
{input}
""")

chinese_only_prompt = ChatPromptTemplate.from_template("""
你是一位專業的技術助理。

請根據下方提供的文件內容，準確回答問題。
若文件中沒有足夠的資訊，請回答：
「在檢索到的文件中找不到足夠的資訊。」

請以**繁體中文**撰寫回答，保持條理清晰。

檢索到的內容：
{context}

問題：
{input}
""")

# ------------------------------------------------------------
# Helper: detect if text is mostly Traditional Chinese
# ------------------------------------------------------------
def is_mostly_chinese(text: str, threshold: float = 0.3) -> bool:
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    ratio = len(chinese_chars) / max(len(text), 1)
    return ratio > threshold


# ------------------------------------------------------------
# Build retrieval chain
# ------------------------------------------------------------
def build_chain(persist_dir: str, collection: str, bilingual: bool = True, k: int = 5):
    vectordb = Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    prompt = bilingual_prompt if bilingual else chinese_only_prompt
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain, retriever


# ------------------------------------------------------------
# Query a collection
# ------------------------------------------------------------
def query_collection(query: str, persist_dir: str, collection: str, k: int = 5):
    print(f"\n🔍 Query: {query}")
    print(f"📚 Using collection: {collection}")

    vectordb = Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # Retrieve documents
    docs = retriever.invoke(query)
    if not docs:
        print("⚠️ No relevant documents retrieved.")
        return "Insufficient information found.", []

    # Display top 3 context previews
    print("\n📄 Retrieved sample context (top 3):\n")
    for i, d in enumerate(docs[:3], start=1):
        meta = d.metadata or {}
        page_no = meta.get("page_number", "N/A")
        print(f"--- Document {i} ---")
        print(f"Source: {meta.get('source', 'unknown')}, Page: {page_no}")
        snippet = d.page_content[:300].replace("\n", " ")
        print(f"Content preview: {snippet}...\n")

    # Determine language composition of the retrieved text
    full_context = " ".join([d.page_content for d in docs])
    bilingual_mode = not is_mostly_chinese(full_context)
    prompt_type = "Bilingual" if bilingual_mode else "Traditional Chinese only"
    print(f"🧠 Detected context language → Using prompt mode: {prompt_type}")

    # Build and invoke chain
    chain, _ = build_chain(persist_dir, collection, bilingual=bilingual_mode, k=k)
    result = chain.invoke({"input": query})

    answer = result.get("answer") or result.get("output_text") or "Insufficient information found."

    print("\n============================")
    print("🎯 Generated Answer:")
    print("============================")
    print(answer)
    return answer, docs


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a single RAG collection with adaptive bilingual output")
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Chroma DB path")
    parser.add_argument("--collection", default="iso_docs", help="Collection name")
    parser.add_argument("--k", type=int, default=5, help="Top-k documents to retrieve")
    args = parser.parse_args()

    query_collection(args.query, args.persist_dir, args.collection, k=args.k)
