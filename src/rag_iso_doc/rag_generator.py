# rag_generator.py (fixed for LC v0.2, Azure, and Chroma)
import argparse
import os

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

# ----------------------- Azure config -----------------------
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_openai_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

missing = [k for k, v in {
    "AZURE_OPENAI_API_KEY": azure_openai_api_key,
    "AZURE_OPENAI_ENDPOINT": azure_openai_endpoint,
    "AZURE_OPENAI_API_VERSION": azure_openai_api_version,
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": azure_openai_chat_deployment,
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": azure_openai_embedding_deployment,
}.items() if not v]
if missing:
    raise ValueError(f"❌ Missing env vars: {', '.join(missing)}")

# ----------------------- LLM & Embeddings -----------------------
llm = AzureChatOpenAI(
    openai_api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    azure_deployment=azure_openai_chat_deployment,
    openai_api_version=azure_openai_api_version,
    # no temperature (Azure model requires default)
)

embeddings = AzureOpenAIEmbeddings(
    openai_api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    azure_deployment=azure_openai_embedding_deployment,
    openai_api_version=azure_openai_api_version,
)

# ----------------------- Prompt -----------------------
prompt = ChatPromptTemplate.from_template("""
You are an assistant specializing in ISO vibration standards.

Use ONLY the retrieved text below to answer the question precisely.
If the retrieved text does not contain enough information, respond:
"Insufficient information found in retrieved documents."

Provide the answer in **Bilingual** format.

**English:**
<English answer>

**繁體中文:**
<Traditional Chinese answer>

Retrieved text:
{context}

Question:
{input}
""")

# ----------------------- Chain builder -----------------------
def build_chain(persist_dir: str, collection: str, k: int = 5):
    vectordb = Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # This chain expects "context" to be a LIST OF Document; it will render them into the {context} slot
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain, retriever

# ----------------------- Run query -----------------------
def generate_answer(query: str, persist_dir: str, collection: str, k: int = 5, show_sources: bool = True):
    print(f"🔍 Using collection: {collection}")
    chain, retriever = build_chain(persist_dir, collection, k=k)

    # Preview retrieved docs (for debugging)
    docs = retriever.invoke(query)
    if not docs:
        print("⚠️ No relevant documents retrieved. Check DB path, collection name, or embeddings consistency.")
        return

    if show_sources:
        print("\n📄 Retrieved sample context (top 2):")
        for i, d in enumerate(docs[:2], start=1):
            meta = d.metadata or {}
            src = meta.get("source", "unknown")
            page = meta.get("page", meta.get("page_number", "N/A"))
            preview = (d.page_content or "").replace("\n", " ")[:300]
            print(f"\n--- Document {i} ---")
            print(f"Source: {src}, Page: {page}")
            print(f"Content preview: {preview}...")

    # Invoke the official retrieval chain (it will pass docs to {context} correctly)
    result = chain.invoke({"input": query})

    # Depending on LC version, the key is usually "answer"
    answer = result.get("answer") or result.get("output_text") or "[No answer generated]"
    print("\n============================")
    print("🎯 Bilingual Answer:")
    print("============================")
    print(answer)

# ----------------------- CLI -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Generator (Azure + Chroma, LC v0.2)")
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Chroma DB path")
    parser.add_argument("--collection", default="iso_docs", help="Chroma collection name (e.g., iso_docs or qa_docs)")
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    parser.add_argument("--no-sources", action="store_true", help="Do not print retrieved source snippets")
    args = parser.parse_args()

    generate_answer(
        query=args.query,
        persist_dir=args.persist_dir,
        collection=args.collection,
        k=args.k,
        show_sources=not args.no_sources,
    )
