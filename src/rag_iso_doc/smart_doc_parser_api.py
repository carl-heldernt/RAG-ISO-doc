import os
import tempfile

import pytesseract
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
load_dotenv()
app = FastAPI(title="Smart Document Parser API")

# ---------------------------------------------------------
# PDF text-extractability check
# ---------------------------------------------------------
def is_pdf_text_extractable(pdf_path: str) -> bool:
    try:
        text = extract_text(pdf_path, maxpages=1)
        return bool(text and text.strip())
    except Exception:
        return False

# ---------------------------------------------------------
# OCR pipeline: pdf2image + pytesseract
# ---------------------------------------------------------
def ocr_pdf_to_documents(pdf_path: str, filename: str) -> list[Document]:
    print(f"📄 OCR processing: {filename}")
    pages = convert_from_path(pdf_path)
    docs = []
    for i, page_img in enumerate(pages, start=1):
        text = pytesseract.image_to_string(page_img, lang="eng")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": filename, "page": i}))
    print(f"✅ OCR extracted {len(docs)} pages from {filename}")
    return docs

# ---------------------------------------------------------
# Token truncation utility
# ---------------------------------------------------------
def truncate_to_token_limit(text: str, model_name: str, max_tokens: int = 12000) -> str:
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        print(f"⚠️ Truncating input from {len(tokens)} → {max_tokens} tokens")
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

# ---------------------------------------------------------
# Endpoint 1: /parse-document
# ---------------------------------------------------------
@app.post("/parse-document")
async def parse_document(file: UploadFile = File(...)):
    """Detects PDF type (text/OCR) → splits into chunks → returns JSON."""
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Step 1: choose parsing method
        if suffix == ".pdf":
            if is_pdf_text_extractable(tmp_path):
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                mode = "text"
                print(f"📘 Parsed {len(docs)} pages from {file.filename} (text mode)")
            else:
                docs = ocr_pdf_to_documents(tmp_path, file.filename)
                mode = "ocr"
        else:
            return JSONResponse({"status": "error", "message": "Only PDF supported."}, status_code=400)

        # Step 2: chunk text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800, chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""], length_function=len,
        )
        chunks = splitter.split_documents(docs)

        results = []
        for idx, doc in enumerate(chunks):
            meta = doc.metadata or {}
            meta["source"] = os.path.basename(file.filename)
            meta.setdefault("page", meta.get("page", idx + 1))
            results.append({"content": doc.page_content, "metadata": meta})

        return JSONResponse({"status": "ok", "mode": mode, "chunks": results, "count": len(results)})

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------------------------------------------------
# Endpoint 2: /parse-llm  (Semantic Parser)
# ---------------------------------------------------------
@app.post("/parse-llm")
async def parse_llm(file: UploadFile = File(...)):
    """Uses Azure OpenAI Chat model to create semantic sections from full text."""
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Load text (use OCR fallback)
        if is_pdf_text_extractable(tmp_path):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            raw_text = "\n".join(d.page_content for d in docs)
            mode = "text"
        else:
            docs = ocr_pdf_to_documents(tmp_path, file.filename)
            raw_text = "\n".join(d.page_content for d in docs)
            mode = "ocr"

        # Initialize Azure Chat LLM
        llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

        # Prompt for semantic segmentation
        prompt = (
            "You are a technical document parser.\n"
            "Split the following text into logical sections with clear titles.\n"
            "For each section, return JSON objects with fields:\n"
            " - section_title\n - content\n - page_range\n - source (use filename)\n"
            "Respond ONLY with valid JSON list."
        )
        # text_input = prompt + "\n\nTEXT:\n" + raw_text[:12000]  # truncate to stay within limits
        model_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        truncated_text = truncate_to_token_limit(raw_text, model_name, max_tokens=12000)
        text_input = prompt + "\n\nTEXT:\n" + truncated_text

        print("🧠 LLM semantic parsing in progress ...")
        response = llm.invoke(text_input)
        output_text = response.content.strip()

        # Validate JSON output
        import json
        try:
            sections = json.loads(output_text)
        except json.JSONDecodeError:
            sections = [{"section_title": "Full Document", "content": raw_text, "page_range": "1-end"}]

        return JSONResponse({
            "status": "ok",
            "mode": mode,
            "sections": sections,
            "count": len(sections),
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
