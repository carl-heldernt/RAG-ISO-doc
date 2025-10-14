import hashlib
import json
import os
import uuid

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

# ---------------------------------------------------------
# Load environment
# ---------------------------------------------------------
load_dotenv()

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

PARSER_URL = os.getenv("SMART_PARSER_API_URL", "http://localhost:8000/parse-document")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def sanitize_metadata(meta: dict) -> dict:
    """
    Convert complex metadata values (dicts, lists, etc.) into JSON strings.
    Keep primitive types as-is (str, int, float, bool, None).
    """
    clean = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        else:
            # JSON-encode complex objects; fallback to str() if JSON fails
            try:
                clean[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                clean[k] = str(v)
    return clean


# ---------------------------------------------------------
# Ingest function
# ---------------------------------------------------------
def ingest_document(file_path: str, persist_dir: str = "./chroma_db", collection_name: str = "iso_docs"):
    print(f"🚀 Sending {file_path} to Smart Parser API at {PARSER_URL} ...")

    with open(file_path, "rb") as f:
        response = requests.post(PARSER_URL, files={"file": f})

    if response.status_code != 200:
        raise RuntimeError(f"❌ Parser API error: {response.status_code} {response.text}")

    data = response.json()
    chunks = data.get("chunks", [])
    print(f"✅ Parser returned {len(chunks)} chunks (mode={data.get('mode')})")

    # filter out empty content & build lists
    texts = []
    metadatas = []
    for c in chunks:
        content = (c.get("content") or "").strip()
        if not content:
            continue
        meta = c.get("metadata", {}) or {}
        # ensure basic metadata exists
        meta.setdefault("source", os.path.basename(file_path))
        # page might be missing or non-int; keep as-is
        texts.append(content)
        metadatas.append(meta)

    if not texts:
        print("⚠️ No textual chunks to ingest; skipping.")
        return

    # Connect to Chroma (will load existing DB if present)
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    # --- Deduplication: compute existing hashes from stored documents ---
    print("🔍 Fetching existing documents for deduplication (this may take time for large DBs)...")
    try:
        existing = vectordb.get()  # includes ids/documents/metadatas by default
        existing_docs = existing.get("documents", []) or []
        existing_ids = set(existing.get("ids", []) or [])
    except Exception as e:
        # If get() fails for some reason, assume empty DB
        print(f"⚠️ Warning: vectordb.get() failed: {e}; assuming empty DB.")
        existing_docs = []
        existing_ids = set()

    existing_hashes = {hash_text(doc) for doc in existing_docs}

    # Prepare new (unique) texts & sanitized metadata & ids
    new_texts = []
    new_metadatas = []
    new_ids = []

    seen_new_hashes = set()
    seen_new_ids = set()
    skipped_count = 0

    for txt, meta in zip(texts, metadatas):
        h = hash_text(txt)

        # Skip duplicates
        if h in existing_hashes or h in seen_new_hashes:
            skipped_count += 1
            continue

        # Avoid ID collision if it somehow exists
        final_id = h
        while final_id in existing_ids or final_id in seen_new_ids:
            final_id = f"{h}_{uuid.uuid4().hex[:6]}"

        new_texts.append(txt)
        # sanitize metadata so Chroma accepts it
        new_metadatas.append(sanitize_metadata(meta))
        # new_ids.append(h)
        new_ids.append(final_id)

        seen_new_hashes.add(h)
        seen_new_ids.add(final_id)

    print(f"🔍 Skipped {skipped_count} duplicate chunks (existing or batch-duplicate).")
    print(f"🆕 {len(new_texts)} unique new chunks to add (out of {len(texts)} parsed).")

    if new_texts:
        try:
            vectordb.add_texts(texts=new_texts, metadatas=new_metadatas, ids=new_ids)
            print(f"💾 Successfully added {len(new_texts)} chunks to collection '{collection_name}' at {persist_dir}")
        except Exception as e:
            print(f"❌ Error while adding texts: {e}")
            raise
    else:
        print("✅ No new content to add; DB unchanged.")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart Document Ingestor (sanitized metadata + dedup)")
    parser.add_argument("--file", required=True, help="Path to the document (PDF, DOCX, etc.)")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Path to Chroma persistence directory")
    parser.add_argument("--collection", default="iso_docs", help="Chroma collection name")
    args = parser.parse_args()

    ingest_document(args.file, args.persist_dir, args.collection)
