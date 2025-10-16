import argparse
import hashlib
import json
import os

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

# =====================================================
# Utility functions
# =====================================================
def hash_id(text: str) -> str:
    """Generate deterministic hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_parser_endpoint(cli_endpoint: str | None = None) -> str:
    """Determine which parser API endpoint to use."""
    return cli_endpoint or os.getenv("SMART_PARSER_ENDPOINT") or "http://localhost:8000/parse-smart"

def build_parser_params(args) -> dict:
    """Build query params for smart_doc_parser_api."""
    params = {}
    if args.ocr_lang:
        params["ocr_lang"] = args.ocr_lang
    if args.diagnostics:
        params["diagnostics"] = "true"
    if args.k_diagram_pages:
        params["k_diagram_pages"] = args.k_diagram_pages
    return params

def fetch_chunks_from_api(file_path: str, parser_endpoint: str, params: dict):
    """Upload document to parser endpoint and get JSON chunks."""
    print(f"📡 Calling parser endpoint: {parser_endpoint}")
    print(f"🧾 Query params: {params}")

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
        resp = requests.post(parser_endpoint, files=files, params=params, timeout=900)
        if resp.status_code != 200:
            raise RuntimeError(f"❌ Parser API error {resp.status_code}: {resp.text}")
        data = resp.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"❌ Parser returned failure: {data}")
        return data["chunks"]

# =====================================================
# Main ingestion function
# =====================================================
def ingest_document(file_path: str, persist_dir: str, collection: str, parser_endpoint: str, params: dict):
    """Send file to smart parser, embed chunks, and store in Chroma."""
    print(f"📘 File: {file_path}")
    chunks = fetch_chunks_from_api(file_path, parser_endpoint, params)
    print(f"✅ Retrieved {len(chunks)} chunks from parser.")

    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_embed = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    azure_ver = os.getenv("AZURE_OPENAI_API_VERSION")

    embeddings = AzureOpenAIEmbeddings(
        openai_api_key=azure_key,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_embed,
        openai_api_version=azure_ver,
    )

    vectordb = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    print(f"📂 Loading existing collection: {collection}")
    try:
        existing = vectordb.get(include=["documents", "metadatas"])
        existing_ids = set(existing.get("ids", []))
    except Exception:
        existing_ids = set()

    new_texts, new_metas, new_ids = [], [], []

    for c in chunks:
        text = c["content"].strip()
        raw_meta = c.get("metadata", {})

        dropped = []
        if isinstance(raw_meta, dict):
            meta = {}
            for k, v in raw_meta.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    meta[k] = v
                else:
                    dropped.append(k)
        else:
            meta, dropped = {}, []

        if dropped:
            print(f"⚠️  Dropped {len(dropped)} unsupported metadata key(s) for this chunk: {', '.join(dropped)}")

        _id = hash_id(text + json.dumps(meta, sort_keys=True))
        if _id not in existing_ids:
            new_texts.append(text)
            new_metas.append(meta)
            new_ids.append(_id)

    if not new_texts:
        print("ℹ️ No new unique chunks found. Skipping ingestion.")
        return

    print(f"🧠 Adding {len(new_texts)} new chunks to '{collection}'...")
    vectordb.add_texts(texts=new_texts, metadatas=new_metas, ids=new_ids)
    print(f"✅ Ingestion complete. Stored at: {persist_dir}")

# =====================================================
# CLI entry point
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest parsed documents into Chroma vector store")

    parser.add_argument("--file", required=True, help="Path to document file (PDF, DOCX, etc.)")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Chroma persistence directory")
    parser.add_argument("--collection", default="iso_docs", help="Collection name in Chroma")
    parser.add_argument("--endpoint", default=None, help="Smart parser endpoint URL (optional)")
    parser.add_argument("--ocr_lang", default=None, help="OCR language(s), e.g. 'eng' or 'eng+chi_tra'")
    parser.add_argument("--diagnostics", action="store_true", help="Enable parser diagnostics mode")
    parser.add_argument("--k_diagram_pages", type=int, default=None, help="Limit of diagram pages for summarization")

    args = parser.parse_args()

    endpoint = get_parser_endpoint(args.endpoint)
    params = build_parser_params(args)

    ingest_document(args.file, args.persist_dir, args.collection, endpoint, params)
