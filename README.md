## Prerequisites packages
- pipx
  - https://pipx.pypa.io/latest/installation/
  - apt install pipx
- UV
  - https://docs.astral.sh/uv/getting-started/installation/#standalone-installer
  - pipx install uv
- Poppler
  - sudo apt install poppler-utils
- Tesseract
  - sudo apt install tesseract-ocr tesseract-ocr-chi-tra

## How to run
```bash
uv run uvicorn rag_demo.smart_doc_parser_api:app \
  --host 0.0.0.0 \
  --port 8000

uv run python src/rag_demo/smart_doc_ingestor.py \
  --file res/UserGuide.pdf \
  --persist_dir ./chroma_db \
  --collection docs_hires \
  --endpoint http://localhost:8000/parse-document \
  --params '{"ocr_lang":"eng+chi_tra"}'
```