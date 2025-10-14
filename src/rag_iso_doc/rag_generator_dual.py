import argparse
import os

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
# Prompt template (bilingual)
# ------------------------------------------------------------
prompt = ChatPromptTemplate.from_template("""
You are an assistant specializing in ISO vibration standards.

Use the retrieved text below to answer the question precisely.
If the retrieved text does not contain enough information, respond:
"Insufficient information found in retrieved documents."

Provide your answer in **bilingual format (English + Traditional Chinese)**.

Retrieved text:
{context}

Question:
{input}
""")

# ------------------------------------------------------------
# Build retrieval chain
# ------------------------------------------------------------
def build_chain(persist_dir: str, collection: str, k: int = 5):
    vectordb = Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain, retriever


# ------------------------------------------------------------
# Run a query against a single collection
# ------------------------------------------------------------
def query_collection(query: str, persist_dir: str, collection: str, k: int = 5):
    print(f"\n📚 Collection: {collection}")
    chain, retriever = build_chain(persist_dir, collection, k)

    docs = retriever.invoke(query)
    if not docs:
        print("⚠️ No relevant documents retrieved.")
        return "Insufficient information found.", []

    # Show retrieved context preview
    for i, d in enumerate(docs[:2], start=1):
        meta = d.metadata or {}
        print(f"\n--- Document {i} ---")
        print(f"Source: {meta.get('source', 'unknown')}, Page: {meta.get('page', meta.get('page_range', 'N/A'))}")
        print(f"Snippet: {d.page_content[:300].replace('\n', ' ')}...")

    result = chain.invoke({"input": query})
    answer = result.get("answer") or result.get("output_text") or "Insufficient information found."
    return answer, docs


# ------------------------------------------------------------
# Compare two collections side-by-side
# ------------------------------------------------------------
def compare_collections(query: str, persist_dir: str, c1: str, c2: str):
    print(f"\n🔍 Query: {query}")

    print("\n==============================")
    print(f"🧩 Classical RAG ({c1})")
    print("==============================")
    ans1, docs1 = query_collection(query, persist_dir, c1)

    print("\n==============================")
    print(f"🧠 LLM-Parsed RAG ({c2})")
    print("==============================")
    ans2, docs2 = query_collection(query, persist_dir, c2)

    print("\n==============================")
    print("🎯 Bilingual Answers Comparison")
    print("==============================")
    print(f"\n--- Classical ({c1}) ---\n{ans1}")
    print(f"\n--- LLM-Parsed ({c2}) ---\n{ans2}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare classical vs LLM-parsed RAG answers")
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Chroma DB path")
    parser.add_argument("--classic", default="iso_docs", help="Classic collection name")
    parser.add_argument("--llm", default="iso_llm_docs", help="LLM-parsed collection name")
    parser.add_argument("--k", type=int, default=5, help="Top-k documents to retrieve")
    args = parser.parse_args()

    compare_collections(args.query, args.persist_dir, args.classic, args.llm)
