# inspect_chunks.py
import json
import os
from textwrap import shorten

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

# ---------------------------------------------------------
# Azure embedding setup
# ---------------------------------------------------------
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

if not all([azure_openai_api_key, azure_openai_endpoint, azure_openai_api_version, azure_openai_embedding_deployment]):
    raise ValueError("❌ Missing Azure OpenAI environment variables. Please check your .env file.")

embeddings = AzureOpenAIEmbeddings(
    openai_api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    azure_deployment=azure_openai_embedding_deployment,
    openai_api_version=azure_openai_api_version,
)

# ---------------------------------------------------------
# Utility: detect and pretty-print JSON values
# ---------------------------------------------------------
def pretty_metadata(meta: dict, indent: int = 2) -> str:
    """Decode JSON-encoded values and return formatted string."""
    pretty = {}
    for k, v in meta.items():
        if isinstance(v, str):
            try:
                # Try decoding JSON string (if stored that way)
                parsed = json.loads(v)
                pretty[k] = parsed
            except Exception:
                pretty[k] = v
        else:
            pretty[k] = v
    return json.dumps(pretty, ensure_ascii=False, indent=indent)


# ---------------------------------------------------------
# Inspect stored chunks
# ---------------------------------------------------------
def inspect_chunks(source_name: str, persist_dir: str = "./chroma_db", collection_name: str = "iso_docs", limit: int = 5):
    """Print all chunks (or sample) for a given source document."""
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    print(f"📂 Inspecting Chroma collection '{collection_name}' (dir={persist_dir})")
    data = vectordb.get()

    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    ids = data.get("ids", [])

    matched = []
    for i, m in enumerate(metas):
        if not m:
            continue
        src = str(m.get("source", "")).lower()
        if source_name.lower() in src:
            matched.append((ids[i], docs[i], m))

    if not matched:
        print(f"⚠️ No chunks found for source containing '{source_name}'")
        return

    print(f"✅ Found {len(matched)} chunks for source '{source_name}'\n")

    for idx, (doc_id, content, meta) in enumerate(matched[:limit]):
        print("=" * 80)
        print(f"🧩 Chunk #{idx+1} | ID: {doc_id}")
        print(f"📄 Content (truncated): {shorten(content, width=300, placeholder='...')}")
        print("📑 Metadata:")
        print(pretty_metadata(meta, indent=4))
        print("=" * 80 + "\n")

    if len(matched) > limit:
        print(f"💡 Showing first {limit} chunks out of {len(matched)} total.\nUse --limit to see more.")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect chunks in Chroma DB for a specific source file")
    parser.add_argument("--source", required=True, help="Substring of source filename (case-insensitive)")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Chroma persistence directory")
    parser.add_argument("--collection", default="iso_docs", help="Chroma collection name")
    parser.add_argument("--limit", type=int, default=5, help="Max number of chunks to display")
    args = parser.parse_args()

    inspect_chunks(args.source, args.persist_dir, args.collection, args.limit)
