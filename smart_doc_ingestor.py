"""
Smart Document Ingestor (for Unstructured Parser API)
-----------------------------------------------------
Uploads a document to the parser API (smart_doc_parser_api.py),
receives page-level merged chunks (text + summarized diagrams),
embeds them using Azure OpenAI embeddings, and stores into Chroma DB.
"""

import argparse
import hashlib
import json
import os

import requests
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import AzureOpenAIEmbeddings


# -----------------------------------------------------
# Utilities
# -----------------------------------------------------
def load_env_var(key: str, required: bool = True) -> str:
    load_dotenv()
    val = os.getenv(key)
    if not val and required:
        raise ValueError(f"❌ Missing environment variable: {key}")
    return val


def compute_id(text: str, source: str, page: int) -> str:
    """Generate deterministic ID (source + page + text hash) for deduplication"""
    h = hashlib.sha256()
    h.update(f"{source}:{page}:{text}".encode("utf-8"))
    return h.hexdigest()


def sanitize_metadata(meta: dict) -> dict:
    """Flatten or stringify complex metadata fields"""
    safe_meta = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe_meta[k] = v
        else:
            safe_meta[k] = str(v)[:500]
    return safe_meta


def call_parser_api(api_url: str, file_path: str, params: dict) -> dict:
    """Upload document to parser API and get structured chunks"""
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
        print(f"📡 Calling parser API: {api_url} with params {params}")
        resp = requests.post(api_url, files=files, params=params, timeout=(300, 1800))
        resp.raise_for_status()
        return resp.json()


# -----------------------------------------------------
# Main ingestion logic
# -----------------------------------------------------
def ingest_document(file_path: str, persist_dir: str, collection: str,
                    parser_endpoint: str, parser_params: dict = None):
    print(f"📤 Uploading '{file_path}' to parser API ...")
    result = call_parser_api(parser_endpoint, file_path, parser_params or {})

    if result.get("status") != "ok":
        raise RuntimeError(f"❌ Parser API returned error: {result.get('message')}")

    chunks = result.get("chunks", [])
    print(f"✅ Retrieved {len(chunks)} merged chunks from parser (mode={result.get('mode')})")

    # --- Initialize embeddings ---
    azure_key = load_env_var("AZURE_OPENAI_API_KEY")
    azure_endpoint = load_env_var("AZURE_OPENAI_ENDPOINT")
    azure_emb_deploy = load_env_var("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

    embeddings = AzureOpenAIEmbeddings(
        openai_api_key=azure_key,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_emb_deploy,
    )

    # --- Initialize Chroma DB ---
    vectordb = Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    try:
        existing_ids = set(vectordb.get().get("ids", []))
    except Exception:
        existing_ids = set()

    # --- Prepare for insertion ---
    new_texts, new_metadatas, new_ids = [], [], []

    for c in chunks:
        text = c.get("content", "").strip()
        if not text:
            continue

        meta = c.get("metadata", {}) or {}
        if isinstance(meta, dict):
            doc_obj = Document(page_content="", metadata=meta)
            try:
                cleaned = filter_complex_metadata(doc_obj)
            except Exception as e:
                print(f"⚠️ filter_complex_metadata failed ({type(e).__name__}): {e}")
                cleaned = doc_obj

            # handle all return types (tuple, dict, Document)
            if isinstance(cleaned, tuple):
                meta_candidates = [v for v in cleaned if isinstance(v, dict)]
                meta = meta_candidates[0] if meta_candidates else meta
            elif hasattr(cleaned, "metadata"):
                meta = cleaned.metadata
            elif isinstance(cleaned, dict):
                meta = cleaned
            else:
                meta = dict(meta)
        else:
            meta = {}

        meta = sanitize_metadata(meta)
        page = meta.get("page_number", -1)
        source = meta.get("source", os.path.basename(file_path))
        uid = compute_id(text, source, page)

        if uid in existing_ids or uid in new_ids:
            print(f"⚠️ Skipping duplicate chunk (page={page}) ID={uid[:12]}")
            continue

        new_texts.append(text)
        new_metadatas.append(meta)
        new_ids.append(uid)

    if not new_texts:
        print("⚠️ No new chunks to add (all duplicates skipped).")
        return

    print(f"💾 Inserting {len(new_texts)} unique page-level chunks into '{collection}' ...")
    vectordb.add_texts(texts=new_texts, metadatas=new_metadatas, ids=new_ids)
    print("✅ Ingestion completed successfully.")


# -----------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Document Ingestor (Page-level)")
    parser.add_argument("--file", required=True, help="Path to document file")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Chroma DB directory")
    parser.add_argument("--collection", default="iso_docs", help="Chroma collection name")
    parser.add_argument("--endpoint", default="http://localhost:8000/parse-document",
                        help="Parser API endpoint URL")
    parser.add_argument("--params", default="{}", help="Optional JSON parameters to send to parser API")

    args = parser.parse_args()
    params = json.loads(args.params)
    ingest_document(args.file, args.persist_dir, args.collection, args.endpoint, params)
