import os
import sys
import re
import logging
from typing import List, Dict, Any

import pdfplumber

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> Dict[str, Any]:
    """
    Opens a PDF and extracts text from every page.

    Returns:
        {
            "filename": str,
            "pages": int,
            "full_text": str,
            "total_words": int
        }
    """
    filename = os.path.basename(file_path)
    print(f"Extracting text from: {filename}")

    try:
        with pdfplumber.open(file_path) as pdf:
            pages = len(pdf.pages)
            page_texts = []

            for i, page in enumerate(pdf.pages):
                raw = page.extract_text()
                if raw:
                    page_texts.append(raw)
                else:
                    logger.warning(f"Page {i+1} of '{filename}' returned no text (may be scanned/image).")

        if not page_texts:
            logger.warning(
                f"No extractable text found in '{filename}'. "
                "This may be a scanned PDF — consider running it through OCR first."
            )
            return {
                "filename": filename,
                "pages": pages,
                "full_text": "",
                "total_words": 0
            }

        # Join all page text
        raw_text = "\n".join(page_texts)

        # --- Text Cleaning ---
        # Fix common encoding artifacts
        cleaned = raw_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        # Collapse multiple spaces/tabs into one
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        # Remove lines that are entirely whitespace/empty
        lines = [ln.strip() for ln in cleaned.splitlines()]
        lines = [ln for ln in lines if ln]
        cleaned = "\n".join(lines)

        total_words = len(cleaned.split())
        print(f"  → Extracted {pages} pages, {total_words} words from '{filename}'")

        return {
            "filename": filename,
            "pages": pages,
            "full_text": cleaned,
            "total_words": total_words
        }

    except Exception as e:
        logger.error(f"Error extracting text from '{filename}': {e}")
        return {
            "filename": filename,
            "pages": 0,
            "full_text": "",
            "total_words": 0,
            "error": str(e)
        }


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into overlapping word-based chunks.

    Args:
        text:       Full document text
        chunk_size: Target number of words per chunk
        overlap:    Number of words to overlap between consecutive chunks

    Returns:
        List of chunk strings
    """
    if not text.strip():
        return []

    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end == len(words):
            break

        start += chunk_size - overlap  # slide forward, keeping overlap

    print(f"  → Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


def process_pdf(file_path: str) -> Dict[str, Any]:
    """
    Full pipeline for a single PDF: extract text then chunk it.

    Returns:
        {
            "filename": str,
            "pages": int,
            "chunks": List[str],
            "total_chunks": int,
            "total_words": int
        }
        On error, includes an "error" key.
    """
    if not os.path.exists(file_path):
        msg = f"File not found: {file_path}"
        logger.error(msg)
        return {"filename": os.path.basename(file_path), "error": msg, "chunks": []}

    if not file_path.lower().endswith(".pdf"):
        msg = f"File is not a PDF: {file_path}"
        logger.error(msg)
        return {"filename": os.path.basename(file_path), "error": msg, "chunks": []}

    extracted = extract_text_from_pdf(file_path)

    if "error" in extracted:
        return {
            "filename": extracted["filename"],
            "pages": extracted.get("pages", 0),
            "chunks": [],
            "total_chunks": 0,
            "total_words": 0,
            "error": extracted["error"]
        }

    if not extracted["full_text"]:
        warning = (
            f"'{extracted['filename']}' appears to be a scanned/image PDF with no extractable text. "
            "Please run OCR on this file before uploading."
        )
        print(f"  ⚠ WARNING: {warning}")
        return {
            "filename": extracted["filename"],
            "pages": extracted["pages"],
            "chunks": [],
            "total_chunks": 0,
            "total_words": 0,
            "warning": warning
        }

    chunks = chunk_text(extracted["full_text"])

    return {
        "filename": extracted["filename"],
        "pages": extracted["pages"],
        "chunks": chunks,
        "total_chunks": len(chunks),
        "total_words": extracted["total_words"]
    }


def process_multiple_pdfs(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Processes a list of PDF file paths one by one.

    Returns:
        List of results from process_pdf() for each file.
    """
    if not file_paths:
        logger.warning("No file paths provided to process_multiple_pdfs.")
        return []

    print(f"\nProcessing {len(file_paths)} PDF(s)...")
    results = []
    for fp in file_paths:
        result = process_pdf(fp)
        results.append(result)

    successful = sum(1 for r in results if "error" not in r and r.get("total_chunks", 0) > 0)
    print(f"Done. {successful}/{len(file_paths)} PDFs processed successfully.\n")
    return results


if __name__ == "__main__":
    print("PDF Extractor ready.")
    print()
    print("Example usage:")
    print("  from pdf_processing.extractor import process_pdf, process_multiple_pdfs")
    print()
    print("  # Single file")
    print("  result = process_pdf('path/to/document.pdf')")
    print("  print(result['total_chunks'], 'chunks extracted')")
    print()
    print("  # Multiple files")
    print("  results = process_multiple_pdfs(['a.pdf', 'b.pdf'])")
    print("  for r in results:")
    print("      print(r['filename'], '->', r.get('total_chunks', 0), 'chunks')")
