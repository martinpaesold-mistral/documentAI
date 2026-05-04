# Document AI

Demo scripts showcasing Mistral's OCR and chat APIs for document understanding.

## Scripts

- **`text_doc_ai.py`** — Extracts structured text, tables, and image bounding boxes from a PDF using Mistral OCR. Outputs results to `output.json`.
- **`qna_doc_ai.py`** — Interactive Q&A over a PDF using Mistral's chat API with document context.
- **`overlay_bboxes.py`** — Draws detected image bounding boxes onto the PDF and saves an annotated copy.

## Setup

```bash
pip install mistralai python-dotenv pymupdf
cp .env.example .env
# Add your Mistral API key to .env
```

## Usage

```bash
# 1. Extract text and image annotations from a PDF
python text_doc_ai.py

# 2. Ask questions about the PDF
python qna_doc_ai.py

# 3. Overlay bounding boxes on the PDF
python overlay_bboxes.py
```
