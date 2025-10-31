import hashlib
import os

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

# --------------------------------------------------------
# Load environment
# --------------------------------------------------------
load_dotenv()

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

if not azure_openai_embedding_deployment:
    raise ValueError("❌ Missing AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME in .env file")

# --------------------------------------------------------
# Configurable parameters
# --------------------------------------------------------
API_URL = "http://localhost:8000/parse-llm"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "iso_llm_docs"

embeddings = AzureOpenAIEmbeddings(
    openai_api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    azure_deployment=azure_openai_embedding_deployment,
    openai_api_version=azure_openai_api_version,
)

# --------------------------------------------------------
# Helper: compute deterministic hash
# --------------------------------------------------------
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# --------------------------------------------------------
# Upload to /parse-llm and retrieve parsed sections
# --------------------------------------------------------
def parse_with_llm(pdf_path: str):
    with open(pdf_path, "rb") as f:
        files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
        response = requests.post(API_URL, files=files)

    if response.status_code != 200:
        raise RuntimeError(f"❌ LLM parser failed: {response.text}")

    result = response.json()
    if result.get("status") != "ok":
        raise RuntimeError(f"⚠️ Parser error: {result.get('message', 'Unknown error')}")
    return result["sections"]

# --------------------------------------------------------
# Ingest into Chroma DB
# --------------------------------------------------------
def ingest_llm_sections(pdf_path: str, persist_dir=PERSIST_DIR, collection=COLLECTION_NAME):
    print(f"📥 Parsing {pdf_path} via LLM API ...")
    sections = parse_with_llm(pdf_path)
    print(f"✅ Received {len(sections)} structured sections from LLM.")

    vectordb = Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    existing_docs = vectordb.get(include=["metadatas"])
    existing_hashes = set()
    for meta in existing_docs.get("metadatas", []):
        if meta and isinstance(meta, dict) and "sha256" in meta:
            existing_hashes.add(meta["sha256"])

    new_texts, new_metadatas, new_ids = [], [], []
    for s in sections:
        content = s.get("content", "").strip()
        if not content:
            continue
        h = hash_text(content)
        if h in existing_hashes:
            continue

        new_texts.append(content)
        new_metadatas.append({
            "source": s.get("source", os.path.basename(pdf_path)),
            "section_title": s.get("section_title", "Untitled"),
            "page_range": s.get("page_range", ""),
            "sha256": h,
        })
        new_ids.append(h)

    if new_texts:
        print(f"🧠 Adding {len(new_texts)} new sections to Chroma DB...")
        vectordb.add_texts(texts=new_texts, metadatas=new_metadatas, ids=new_ids)
    else:
        print("ℹ️ No new sections to add (all duplicates).")

    print("💾 Ingestion complete.")
    return len(new_texts)

# --------------------------------------------------------
# CLI entry point
# --------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest LLM-parsed document sections into Chroma.")
    parser.add_argument("--file", required=True, help="Path to PDF file.")
    parser.add_argument("--persist_dir", default=PERSIST_DIR, help="Path to Chroma persistence directory.")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Chroma collection name.")
    args = parser.parse_args()

    ingest_llm_sections(args.file, args.persist_dir, args.collection)
