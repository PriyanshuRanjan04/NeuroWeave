import pdfplumber
import logging
from typing import List, Dict, Any
import os
import sys

# Add project root to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import clean_text, chunk_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text from a single PDF file, clean it, and chunk it.
    Returns structured data: {filename, pages, chunks, total_words, error(optional), warning(optional)}
    """
    filename = os.path.basename(pdf_path)
    result = {
        "filename": filename,
        "pages": 0,
        "chunks": [],
        "total_words": 0
    }
    
    try:
        full_text = []
        with pdfplumber.open(pdf_path) as pdf:
            result["pages"] = len(pdf.pages)
            
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    cleaned_page_text = clean_text(page_text)
                    full_text.append(cleaned_page_text)
                    
        combined_text = " ".join(full_text)
        result["total_words"] = len(combined_text.split())
        
        if result["total_words"] == 0 and result["pages"] > 0:
            logger.warning(f"Scanned PDF detected or no text found for {filename}.")
            result["warning"] = "Scanned PDF detected or no text found. OCR might be needed."
            
        result["chunks"] = chunk_text(combined_text, chunk_size=1000, overlap=200)
        logger.info(f"Successfully processed {filename}. Pages: {result['pages']}, Chunks: {len(result['chunks'])}")
        
    except Exception as e:
        logger.error(f"Failed to process PDF {filename}: {str(e)}")
        result["error"] = f"Corrupted or invalid PDF: {str(e)}"
        
    return result

def process_multiple_pdfs(pdf_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Process multiple PDFs and return a list of structured data representations.
    """
    results = []
    for path in pdf_paths:
        if os.path.exists(path):
            results.append(extract_text_from_pdf(path))
        else:
            logger.error(f"File not found: {path}")
            results.append({
                "filename": os.path.basename(path) if hasattr(os.path, 'basename') else str(path),
                "error": "File not found."
            })
    return results
