"""
Inspect Chroma Collection Chunks
--------------------------------
Analyzes and summarizes stored chunks from Chroma DB:
 - Displays per-document statistics
 - Shows page-level summaries and metadata
 - Counts text, diagrams, and average lengths
"""

import argparse
import os
import textwrap

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings


def load_env_var(key: str, required: bool = True) -> str:
    load_dotenv()
    val = os.getenv(key)
    if not val and required:
        raise ValueError(f"❌ Missing environment variable: {key}")
    return val


def preview_text(text: str, width=80, max_lines=5):
    wrapped = textwrap.fill(text.strip().replace("\n", " "), width)
    lines = wrapped.split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (truncated) ..."]
    return "\n".join(lines)


def inspect_collection(persist_dir: str, collection: str):
    azure_key = load_env_var("AZURE_OPENAI_API_KEY")
    azure_endpoint = load_env_var("AZURE_OPENAI_ENDPOINT")
    azure_emb_deploy = load_env_var("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

    embeddings = AzureOpenAIEmbeddings(
        openai_api_key=azure_key,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_emb_deploy,
    )

    vectordb = Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    data = vectordb.get(include=["metadatas", "documents"])
    if not data["documents"]:
        print("⚠️ No documents found in collection.")
        return

    # -------------------------------
    # Aggregate stats
    # -------------------------------
    doc_count = len(data["documents"])
    avg_len = sum(len(t) for t in data["documents"]) / doc_count
    pages_with_visuals = sum(
        1 for m in data["metadatas"] if "Table" in str(m) or "Image" in str(m)
    )

    print(f"\n📚 Collection: {collection}")
    print(f"📁 Persist dir: {persist_dir}")
    print(f"🧩 Total chunks: {doc_count}")
    print(f"🧮 Avg length: {avg_len:.1f} chars")
    print(f"🖼️ Chunks containing diagrams/tables: {pages_with_visuals}")
    print("=" * 80)

    # -------------------------------
    # Group by source and page
    # -------------------------------
    grouped = {}
    for text, meta in zip(data["documents"], data["metadatas"]):
        src = meta.get("source", "unknown")
        pg = meta.get("page_number", -1)
        grouped.setdefault(src, {}).setdefault(pg, []).append((text, meta))

    for src, pages in grouped.items():
        print(f"\n📘 Document: {src} ({len(pages)} pages)")
        print("-" * 80)
        for page, chunks in sorted(pages.items()):
            combined = "\n".join(t for t, _ in chunks)
            meta = chunks[0][1]
            types = meta.get("element_types", [])
            print(f"\n📄 Page {page} | Types: {types}")
            print(preview_text(combined))
            print("-" * 80)

    print("\n✅ Inspection completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Chroma Collection Chunks")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Path to Chroma DB directory")
    parser.add_argument("--collection", default="iso_docs", help="Collection name to inspect")
    args = parser.parse_args()

    inspect_collection(args.persist_dir, args.collection)
