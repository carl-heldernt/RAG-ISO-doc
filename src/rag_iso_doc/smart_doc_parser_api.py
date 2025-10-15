import json
import os
import tempfile

import pytesseract
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredEPubLoader, UnstructuredExcelLoader,
    UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader)
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text

# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------
load_dotenv()
app = FastAPI(title="Universal Smart Document Parser API (Hybrid PDF)")

# ---------------------------------------------------------
# Helper: token truncation
# ---------------------------------------------------------
def truncate_to_token_limit(text: str, model_name: str, max_tokens: int = 12000) -> str:
    """Safely truncate text to fit model token window."""
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        print(f"⚠️ Truncating from {len(tokens)} → {max_tokens} tokens")
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

# ---------------------------------------------------------
# Helper: detect if PDF has extractable text
# ---------------------------------------------------------
def is_pdf_text_extractable(pdf_path: str) -> bool:
    try:
        text = extract_text(pdf_path, maxpages=1)
        return bool(text and text.strip())
    except Exception:
        return False

# ---------------------------------------------------------
# Helper: OCR per page
# ---------------------------------------------------------
def ocr_pdf_to_documents(pdf_path: str, filename: str) -> list[Document]:
    print(f"📄 OCR processing: {filename}")
    pages = convert_from_path(pdf_path)
    docs = []
    for i, page_img in enumerate(pages, start=1):
        text = pytesseract.image_to_string(page_img, lang="eng")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": filename, "page": i, "mode": "ocr"}))
    print(f"✅ OCR extracted {len(docs)} pages from {filename}")
    return docs

# ---------------------------------------------------------
# Helper: Hybrid PDF Extractor (merge text + OCR)
# ---------------------------------------------------------
def hybrid_extract_pdf(pdf_path: str, filename: str) -> list[Document]:
    """Combine PyPDFLoader text and OCR text per page."""
    text_docs = []
    ocr_docs = []
    try:
        loader = PyPDFLoader(pdf_path)
        text_docs = loader.load()
    except Exception:
        pass

    ocr_docs = ocr_pdf_to_documents(pdf_path, filename)
    merged_docs = []

    for i in range(1, max(len(text_docs), len(ocr_docs)) + 1):
        text_part = next((d.page_content for d in text_docs if d.metadata.get("page") == i), "")
        ocr_part = next((d.page_content for d in ocr_docs if d.metadata.get("page") == i), "")
        merged_text = (text_part + "\n" + ocr_part).strip()
        if merged_text:
            merged_docs.append(
                Document(
                    page_content=merged_text,
                    metadata={"source": filename, "page": i, "mode": "text+ocr"},
                )
            )

    return merged_docs

# ---------------------------------------------------------
# Helper: Generic document loader
# ---------------------------------------------------------
def load_document_generic(filepath: str, filename: str) -> list[Document]:
    """Auto-detect document type, with hybrid PDF and fallback loader."""
    suffix = os.path.splitext(filename)[1].lower()

    try:
        if suffix == ".pdf":
            if is_pdf_text_extractable(filepath):
                docs = hybrid_extract_pdf(filepath, filename)
            else:
                docs = ocr_pdf_to_documents(filepath, filename)
        elif suffix in [".doc", ".docx"]:
            loader = UnstructuredWordDocumentLoader(filepath)
            docs = loader.load()
        elif suffix in [".xls", ".xlsx"]:
            loader = UnstructuredExcelLoader(filepath)
            docs = loader.load()
        elif suffix in [".ppt", ".pptx"]:
            loader = UnstructuredPowerPointLoader(filepath)
            docs = loader.load()
        elif suffix in [".md"]:
            loader = UnstructuredMarkdownLoader(filepath)
            docs = loader.load()
        elif suffix in [".html", ".htm"]:
            loader = UnstructuredHTMLLoader(filepath)
            docs = loader.load()
        elif suffix in [".epub"]:
            loader = UnstructuredEPubLoader(filepath)
            docs = loader.load()
        elif suffix in [".txt", ".csv", ".log"]:
            loader = UnstructuredFileLoader(filepath)
            docs = loader.load()
        else:
            print(f"⚠️ Unknown file extension '{suffix}', using UnstructuredFileLoader fallback.")
            loader = UnstructuredFileLoader(filepath)
            docs = loader.load()

        for d in docs:
            d.metadata["source"] = filename
        return docs

    except Exception as e:
        print(f"❌ Error loading {filename}: {e}. Falling back to UnstructuredFileLoader.")
        loader = UnstructuredFileLoader(filepath)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = filename
        return docs

# ---------------------------------------------------------
# Endpoint 1: /parse-document
# ---------------------------------------------------------
@app.post("/parse-document")
async def parse_document(file: UploadFile = File(...)):
    """Generic file parsing + hybrid PDF support + chunking."""
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        docs = load_document_generic(tmp_path, file.filename)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(docs)

        results = []
        for idx, doc in enumerate(chunks):
            meta = doc.metadata or {}
            meta["source"] = os.path.basename(file.filename)
            meta.setdefault("page", meta.get("page", idx + 1))
            results.append({"content": doc.page_content, "metadata": meta})

        return JSONResponse({"status": "ok", "chunks": results, "count": len(results)})

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------------------------------------------------
# Endpoint 2: /parse-llm (Semantic Parser)
# ---------------------------------------------------------
@app.post("/parse-llm")
async def parse_llm(file: UploadFile = File(...)):
    """Use Azure OpenAI to semantically segment any document."""
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        docs = load_document_generic(tmp_path, file.filename)
        raw_text = "\n".join(d.page_content for d in docs)

        llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

        prompt = (
            "You are a semantic document parser.\n"
            "Divide the text into logical sections with titles.\n"
            "Output JSON array with:\n"
            "  - section_title\n  - content\n  - page_range (if known)\n  - source (filename)\n"
            "Respond ONLY with valid JSON."
        )

        model_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o")
        truncated_text = truncate_to_token_limit(raw_text, model_name, max_tokens=12000)
        text_input = prompt + "\n\nTEXT:\n" + truncated_text

        print("🧠 LLM semantic parsing in progress ...")
        response = llm.invoke(text_input)
        output_text = response.content.strip()

        try:
            sections = json.loads(output_text)
        except json.JSONDecodeError:
            sections = [{"section_title": "Full Document", "content": truncated_text, "page_range": "1-end"}]

        return JSONResponse({"status": "ok", "mode": "llm", "sections": sections, "count": len(sections)})

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
