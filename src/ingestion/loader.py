import logging
from pathlib import Path
import pymupdf
from tqdm import tqdm
from configs.config import RAW_DOCS_PATH, LOG_LEVEL


logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)



def _detect_document_type(filename: str) -> str:
    """
    Detects document type from filename.
    Private helper function — only used inside loader.py
    """
    filename_lower = filename.lower()

    if "annual" in filename_lower:
        return "annual_report"
    elif "sustainability" in filename_lower:
        return "sustainability_report"
    elif "cmd" in filename_lower or "capital" in filename_lower:
        return "capital_markets"
    elif "whitepaper" in filename_lower or "white" in filename_lower:
        return "whitepaper"
    elif "mtm" in filename_lower or "management" in filename_lower:
        return "management_presentation"
    else:
        return "technical_document"



def extract_text_from_pdf(pdf_path: str):
    """
    Opens one PDF and extracts text from each page.
    Returns list of page dictionaries with text and metadata.
    """
    pages = []
    filename = Path(pdf_path).name

    try:
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)

        logger.info(f"Processing: {filename} ({total_pages} pages)")

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()

            if len(text.strip()) < 50:
                logger.debug(
                    f"Skipping page {page_num + 1} — insufficient text"
                )
                continue

            page_data = {
                "text": text.strip(),
                "page_number": page_num + 1,
                "source": filename,
                "total_pages": total_pages,
                "document_type": _detect_document_type(filename),
            }
            pages.append(page_data)

        doc.close()
        logger.info(f"Extracted {len(pages)} pages from {filename}")

    except Exception as e:
        logger.error(f"Failed to process {filename}: {e}")

    return pages




def load_documents(docs_path: str = RAW_DOCS_PATH):
    """
    Finds all PDF files in folder and extracts text page by page.
    Returns list of page dictionaries with text and metadata.
    """
    docs_folder = Path(docs_path)

    if not docs_folder.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {docs_path}\n"
            f"Please add PDF files to {docs_path}"
        )

    pdf_files = list(docs_folder.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {docs_path}")

    logger.info(f"Found {len(pdf_files)} PDF files in {docs_path}")

    all_pages = []

    for pdf_path in tqdm(pdf_files, desc="Loading documents"):
        pages = extract_text_from_pdf(str(pdf_path))
        all_pages.extend(pages)

    logger.info(f"Total pages extracted: {len(all_pages)}")

    return all_pages

