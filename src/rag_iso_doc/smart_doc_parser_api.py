import json
import os
import re
import tempfile
from datetime import datetime
from statistics import mean

import pytesseract
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredEPubLoader, UnstructuredExcelLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader)
# LLM
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Loaders (updated imports per LangChain 0.2.8+)
from langchain_unstructured import UnstructuredLoader
from pdf2image import convert_from_path
# PDF text/ocr deps
from pdfminer.high_level import extract_text

# =========================
# App / Env
# =========================
load_dotenv()
app = FastAPI(title="Smart Parser API (Hybrid + Diagram Summarizer + Diagnostics)")

# =========================
# Token truncation helper
# =========================
def truncate_to_token_limit(text: str, model_name: str, max_tokens: int = 12000) -> str:
    """Safely truncate long text to fit within model's context."""
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        print(f"⚠️ Truncating from {len(tokens)} → {max_tokens} tokens")
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

# =========================
# PDF helpers
# =========================
def is_pdf_text_extractable(pdf_path: str) -> bool:
    try:
        text = extract_text(pdf_path, maxpages=1)
        return bool(text and text.strip())
    except Exception:
        return False

def ocr_pdf_to_documents(pdf_path: str, filename: str, ocr_lang: str = "eng") -> list[Document]:
    """Per-page OCR to Documents with metadata (page, source)."""
    pages = convert_from_path(pdf_path)
    docs = []
    for i, page_img in enumerate(pages, start=1):
        text = pytesseract.image_to_string(page_img, lang=ocr_lang)
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": filename, "page": i, "mode": "ocr"}))
    return docs

def hybrid_extract_pdf(pdf_path: str, filename: str, ocr_lang: str = "eng") -> list[Document]:
    """Combine extractable text + OCR for full coverage."""
    text_docs = []
    try:
        loader = PyPDFLoader(pdf_path)
        text_docs = loader.load()
    except Exception:
        pass

    ocr_docs = ocr_pdf_to_documents(pdf_path, filename, ocr_lang=ocr_lang)
    merged_docs = []
    max_pages = max(len(text_docs), len(ocr_docs))
    for i in range(1, max_pages + 1):
        text_part = next((d.page_content for d in text_docs if d.metadata.get("page") == i), "")
        ocr_part = next((d.page_content for d in ocr_docs if d.metadata.get("page") == i), "")
        merged_text = (text_part + "\n" + ocr_part).strip()
        if merged_text:
            merged_docs.append(
                Document(
                    page_content=merged_text,
                    metadata={
                        "source": filename,
                        "page": i,
                        "mode": "text+ocr" if text_part and ocr_part else ("text" if text_part else "ocr"),
                    },
                )
            )
    return merged_docs or text_docs or ocr_docs

# =========================
# Generic multi-format loader
# =========================
def load_document_generic(filepath: str, filename: str, ocr_lang: str = "eng") -> list[Document]:
    suffix = os.path.splitext(filename)[1].lower()

    try:
        if suffix == ".pdf":
            if is_pdf_text_extractable(filepath):
                docs = hybrid_extract_pdf(filepath, filename, ocr_lang=ocr_lang)
            else:
                docs = ocr_pdf_to_documents(filepath, filename, ocr_lang=ocr_lang)
        elif suffix in [".doc", ".docx"]:
            docs = UnstructuredWordDocumentLoader(filepath).load()
        elif suffix in [".xls", ".xlsx"]:
            docs = UnstructuredExcelLoader(filepath).load()
        elif suffix in [".ppt", ".pptx"]:
            docs = UnstructuredPowerPointLoader(filepath).load()
        elif suffix in [".md"]:
            docs = UnstructuredMarkdownLoader(filepath).load()
        elif suffix in [".html", ".htm"]:
            docs = UnstructuredHTMLLoader(filepath).load()
        elif suffix in [".epub"]:
            docs = UnstructuredEPubLoader(filepath).load()
        elif suffix in [".txt", ".csv", ".json", ".yaml", ".yml", ".rst", ".log"]:
            docs = UnstructuredLoader(filepath).load()
        else:
            print(f"⚠️ Unknown extension '{suffix}', using UnstructuredLoader fallback.")
            docs = UnstructuredLoader(filepath).load()

        for d in docs:
            d.metadata["source"] = filename
        return docs

    except Exception as e:
        print(f"❌ Error loading {filename}: {e}, fallback to UnstructuredLoader.")
        docs = UnstructuredLoader(filepath).load()
        for d in docs:
            d.metadata["source"] = filename
        return docs

# =========================
# Diagram heuristics
# =========================
def looks_like_diagram_tokens(text: str) -> bool:
    """Detect scattered OCR text typical of diagrams/UI pages."""
    if not text or len(text) < 20:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    avg_len = mean(len(ln) for ln in lines)
    short_lines = sum(1 for ln in lines if len(ln) <= 12)
    title_like = sum(1 for ln in lines if re.match(r"^[A-Z][A-Za-z0-9\-/ ]{0,20}$", ln))
    ui_words = {"ok", "next", "cancel", "apply", "save", "start", "stop", "dispatch", "menu"}
    ui_hits = sum(1 for ln in lines if ln.lower() in ui_words)
    return (avg_len < 18 and (short_lines / len(lines) > 0.5 or title_like / len(lines) > 0.35 or ui_hits >= 2))

# =========================
# Diagram summarization
# =========================
def summarize_diagram_fragments(fragments_text: str, filename: str) -> str:
    """Summarize diagram tokens into coherent natural language using AzureChatOpenAI."""
    llm = AzureChatOpenAI(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    prompt = (
        "You are a document visual interpreter.\n"
        "You will receive short text fragments extracted from diagrams or UI screenshots.\n"
        "Summarize what the diagram represents in clear, neutral language. "
        "If fragments look like UI buttons, steps, or labels, infer their purpose and logical sequence.\n\n"
        f"FRAGMENTS (from {filename}):\n{fragments_text}"
    )
    model_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o")
    safe_prompt = truncate_to_token_limit(prompt, model_name, max_tokens=3000)
    resp = llm.invoke(safe_prompt)
    return resp.content.strip()

# =========================
# Unified smart endpoint (with diagnostics + file export)
# =========================
@app.post("/parse-smart")
async def parse_smart(
    file: UploadFile = File(...),
    ocr_lang: str = "eng",
    k_diagram_pages: int = 9999,
    diagnostics: bool = False,
):
    """
    Smart parser for PDFs, Word, Excel, PPT, HTML, etc.
    - Hybrid PDF text+OCR extraction
    - Diagram detection + LLM summarization
    - Diagnostics: shows & saves OCR vs LLM summaries
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        docs = load_document_generic(tmp_path, file.filename, ocr_lang=ocr_lang)
        pages_summarized = 0
        diagnostics_log = []
        enriched_docs = []

        for d in docs:
            text = d.page_content or ""
            if d.metadata.get("page") and looks_like_diagram_tokens(text) and pages_summarized < k_diagram_pages:
                summary = summarize_diagram_fragments(text, d.metadata.get("source", file.filename))
                enriched_docs.append(
                    Document(
                        page_content=summary,
                        metadata={**d.metadata, "diagram_summary": True, "summarization_model": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")},
                    )
                )
                if diagnostics:
                    sample_lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:8]
                    diagnostics_log.append({
                        "page": d.metadata.get("page"),
                        "source": d.metadata.get("source", file.filename),
                        "original_fragments": sample_lines,
                        "llm_summary": summary,
                    })
                pages_summarized += 1
            else:
                enriched_docs.append(d)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=200, separators=["\n\n", "\n", " ", ""], length_function=len)
        chunks = splitter.split_documents(enriched_docs)

        results = []
        for idx, doc in enumerate(chunks):
            meta = doc.metadata or {}
            meta["source"] = os.path.basename(file.filename)
            meta.setdefault("page", meta.get("page", idx + 1))
            results.append({"content": doc.page_content, "metadata": meta})

        response = {
            "status": "ok",
            "mode": "smart",
            "count": len(results),
            "diagram_pages_summarized": pages_summarized,
            "chunks": results,
        }

        # 💾 Diagnostics file export
        if diagnostics and diagnostics_log:
            diag_dir = os.path.join(os.getcwd(), "diagnostics")
            os.makedirs(diag_dir, exist_ok=True)
            diag_path = os.path.join(diag_dir, f"{os.path.splitext(file.filename)[0]}_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(diagnostics_log, f, ensure_ascii=False, indent=2)
            response["diagnostics"] = diagnostics_log
            response["diagnostics_file"] = diag_path

        return JSONResponse(response)

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
