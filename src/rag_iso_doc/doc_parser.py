# doc_parser.py
import argparse
import hashlib
import os

import pytesseract
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
# NOTE: we no longer use UnstructuredPDFLoader for OCR fallback (we do per-page OCR)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path

load_dotenv()

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
OCR_LANG = os.getenv("OCR_LANG", "eng")   # e.g. "eng" or "chi_tra"

if not azure_openai_api_key:
    raise ValueError("❌ AZURE_OPENAI_API_KEY not found in .env")
if not azure_openai_embedding_deployment:
    raise ValueError("❌ AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME not found in .env")


def load_pdf_with_fallback(pdf_path: str, dpi: int = 200) -> list[Document]:
    """
    Try to extract text using PyPDFLoader (per-page). If there is no extractable text,
    perform per-page OCR using pdf2image + pytesseract and return one Document per page
    with metadata['page'] set correctly.
    """
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # typically returns one Document per page when text layer exists

    # If any page has text content, treat as text-based PDF
    has_text = any(page.page_content and page.page_content.strip() for page in pages)
    if has_text:
        print("✅ Extractable text detected. Returning page Documents from PyPDFLoader.")
        # Ensure 'page' metadata exists (PyPDFLoader often provides page # in metadata)
        for i, p in enumerate(pages, start=1):
            if "page" not in p.metadata:
                p.metadata["page"] = i
        return pages

    # Otherwise: PDF likely image-only -> perform per-page OCR
    print("⚠️ No extractable text found. Performing per-page OCR with pdf2image + pytesseract...")
    images = convert_from_path(pdf_path, dpi=dpi)  # requires poppler
    ocr_docs: list[Document] = []
    for i, img in enumerate(images, start=1):
        # pytesseract returns a string; choose OCR_LANG via env var if needed
        try:
            text = pytesseract.image_to_string(img, lang=OCR_LANG)
        except Exception as e:
            # fallback to default language if specified lang not installed
            print(f"⚠️ pytesseract error on page {i}: {e}. Retrying with default config...")
            text = pytesseract.image_to_string(img)

        # Basic cleanup: normalize whitespace
        if text:
            text = " ".join(text.split())

        doc = Document(
            page_content=text or "",
            metadata={
                "source": os.path.basename(pdf_path),
                "page": i
            }
        )
        ocr_docs.append(doc)

    print(f"🖨️ OCR produced {len(ocr_docs)} page-documents.")
    return ocr_docs


def chunk_documents(docs, chunk_size=1200, overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def hash_content(text: str) -> str:
    """Generate hash from normalized text."""
    norm = " ".join(text.split())  # collapse whitespace
    return hashlib.md5(norm.lower().encode("utf-8")).hexdigest()


def build_vectorstore(pdf_paths, persist_dir: str):
    embeddings = AzureOpenAIEmbeddings(
        openai_api_key=azure_openai_api_key,
        azure_deployment=azure_openai_embedding_deployment,
    )

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    all_new_chunks = []
    for pdf_path in pdf_paths:
        docs = load_pdf_with_fallback(pdf_path)

        for d in docs:
            if "source" not in d.metadata or not d.metadata["source"]:
                d.metadata["source"] = os.path.basename(pdf_path)
        print(f"📄 Loaded {len(docs)} page-docs from {pdf_path}")

        chunks = chunk_documents(docs)
        print(f"✂️ Split {pdf_path} into {len(chunks)} chunks.")

        # assign a stable id = source + page + chunk index
        for idx, c in enumerate(chunks):
            source = c.metadata.get("source", os.path.basename(pdf_path))
            page = c.metadata.get("page", 0)
            stable_id = f"{source}_p{page}_c{idx}"
            c.metadata["doc_id"] = stable_id
        all_new_chunks.extend(chunks)

    if not all_new_chunks:
        print("⚠️ No chunks generated; nothing to add.")
        return vectordb

    # Deduplication: get existing IDs (if any)
    try:
        existing = vectordb.get()  # returns {"ids": [...], "documents": [...], "metadatas": [...] }
        existing_ids = set(existing.get("ids", []))
    except Exception:
        existing_ids = set()

    unique_chunks = [c for c in all_new_chunks if c.metadata["doc_id"] not in existing_ids]

    if unique_chunks:
        ids = [c.metadata["doc_id"] for c in unique_chunks]
        vectordb.add_documents(unique_chunks, ids=ids)
        print(f"✅ Added {len(unique_chunks)} new unique chunks.")
    else:
        print("ℹ️ All chunks already exist in the DB; no new additions.")

    return vectordb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PDF(s), chunk, embed, and store in Chroma DB (deduped).")
    parser.add_argument("pdfs", nargs="+", help="Path(s) to one or more PDF files to process")
    parser.add_argument("--persist_dir", default="chroma_iso10816", help="Directory for Chroma persistence")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF->image conversion (OCR fallback)")
    args = parser.parse_args()

    build_vectorstore(args.pdfs, args.persist_dir)
