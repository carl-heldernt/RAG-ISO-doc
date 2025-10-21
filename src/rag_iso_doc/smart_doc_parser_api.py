"""
Smart Document Parser API
--------------------------
Enhanced with:
 - Streaming OCR progress updates (FastAPI StreamingResponse)
 - Token-safe truncation (tiktoken-based)
 - Unstructured PDF/Doc/Excel parsing with LLM diagram summarization
"""

import os
import tempfile
import warnings

import pytesseract
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from unstructured.documents.elements import (Image, ListItem, NarrativeText,
                                             Table, Title)
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_GPU"] = "false"
os.environ["UNSTRUCTURED_USE_GPU"] = "false"

warnings.filterwarnings("ignore", message="The `max_size` parameter is deprecated")

app = FastAPI(title="Smart Document Parser API (Streaming + TokenSafe)")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

llm = AzureChatOpenAI(
    azure_deployment=AZURE_CHAT_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
)

# -----------------------------------------------------
# Utilities
# -----------------------------------------------------
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


def is_pdf_text_extractable(pdf_path: str) -> bool:
    """Detect if PDF contains extractable text."""
    try:
        text = extract_text(pdf_path, maxpages=1)
        return bool(text and text.strip())
    except Exception:
        return False


def summarize_diagram_fragments(raw_text: str) -> str:
    """Use LLM to summarize OCR fragments from diagrams."""
    if not raw_text.strip():
        return ""
    prompt = (
        "The following OCR text is extracted from a diagram or figure. "
        "Summarize it into a short, coherent paragraph while preserving key information.\n\n"
        f"TEXT:\n{truncate_to_token_limit(raw_text, llm.model_name)}"
    )
    return llm.invoke(prompt).content.strip()


def ocr_pdf_to_text_stream(pdf_path: str, lang: str = "eng"):
    """Yield OCR text page by page, with progress updates."""
    pages = convert_from_path(pdf_path)
    total = len(pages)
    yield f"Starting OCR for {total} pages (lang={lang})...\n"

    for i, page in enumerate(pages, start=1):
        text = pytesseract.image_to_string(page, lang=lang)
        yield f"[Page {i}/{total}] Done\n"
        yield f"--- Page {i} ---\n{text.strip()}\n"

    yield "✅ OCR completed successfully.\n"


# -----------------------------------------------------
# Main endpoint: parse any document
# -----------------------------------------------------
@app.post("/parse-document")
async def parse_document(request: Request, file: UploadFile = File(...)):
    """
    Parse document using Unstructured + OCR + LLM summarization.
    Stream OCR progress to client in real-time.
    """
    mode = request.query_params.get("mode", "hi_res")
    ocr_lang = request.query_params.get("ocr_lang", "eng")
    stream = request.query_params.get("stream", "false").lower() == "true"

    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            if is_pdf_text_extractable(tmp_path):
                print(f"📄 Using partition_pdf (strategy={mode}) for text/mixed PDF...")
                try:
                    elements = partition_pdf(
                        filename=tmp_path,
                        strategy=mode,
                        languages=[ocr_lang],
                        infer_table_structure=True,
                        extract_image_block_types=["Image", "Table"],
                        extract_image_block_to_payload=True,
                    )
                except Exception as e:
                    print(f"❌ partition_pdf(hi_res) failed: {e}")
                    return JSONResponse(
                        {"status": "error", "message": f"hi_res parsing failed: {e}"},
                        status_code=500
                    )
            else:
                print(f"🧠 Non-extractable PDF → OCR mode ({ocr_lang})")
                if stream:
                    # Stream OCR progress
                    return StreamingResponse(
                        ocr_pdf_to_text_stream(tmp_path, lang=ocr_lang),
                        media_type="text/plain"
                    )
                else:
                    # Regular OCR (non-stream)
                    pages = convert_from_path(tmp_path)
                    total = len(pages)
                    elements = []
                    for i, page in enumerate(pages, start=1):
                        print(f"🔍 OCR page {i}/{total}")
                        text = pytesseract.image_to_string(page, lang=ocr_lang)
                        if text.strip():
                            el = NarrativeText(text=text.strip())
                            # ⚡ Assign page metadata so downstream bundling works
                            el.metadata.page_number = i
                            elements.append(el)
                    mode = "ocr_pdf"
        else:
            print(f"📄 Using generic Unstructured parser for {suffix} document...")
            elements = partition(filename=tmp_path)
            mode = "generic"

        # --- Step 2: Page-level aggregation ---
        page_bundles = {}
        for el in elements:
            pg = getattr(el.metadata, "page_number", 1)
            page_bundles.setdefault(pg, {"texts": [], "types": []})
            if isinstance(el, (NarrativeText, Title, ListItem)):
                page_bundles[pg]["texts"].append(el.text)
                page_bundles[pg]["types"].append(el.__class__.__name__)
            elif isinstance(el, (Image, Table)):
                desc = summarize_diagram_fragments(el.text or "")
                page_bundles[pg]["texts"].append(desc)
                page_bundles[pg]["types"].append(el.__class__.__name__)

        chunks = []
        for pg, data in sorted(page_bundles.items()):
            merged_text = "\n".join([t for t in data["texts"] if t.strip()])
            chunks.append({
                "content": merged_text.strip(),
                "metadata": {
                    "source": file.filename,
                    "page_number": pg,
                    "element_types": list(set(data["types"])),
                },
            })

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        docs = [type("TempDoc", (), {"page_content": c["content"], "metadata": c["metadata"]}) for c in chunks]
        split_docs = splitter.split_documents(docs)
        results = [{"content": d.page_content, "metadata": d.metadata} for d in split_docs]

        return JSONResponse({"status": "ok", "mode": mode, "count": len(results), "chunks": results})

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
