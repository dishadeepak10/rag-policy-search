import fitz  # pymupdf
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os

# Point pytesseract to the Tesseract install we confirmed earlier
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF, page by page.
    Tries direct text extraction first; falls back to OCR if a page
    has no extractable text (i.e., it's a scanned image).
    Returns a list of dicts: [{"page": 1, "text": "..."}, ...]
    """
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if text:
            # Native digital text found
            pages_data.append({"page": page_num + 1, "text": text})
        else:
            # No text found -> likely a scanned/image page -> run OCR
            print(f"Page {page_num + 1} of {os.path.basename(pdf_path)} has no text — running OCR...")
            ocr_text = ocr_page(pdf_path, page_num)
            pages_data.append({"page": page_num + 1, "text": ocr_text})

    doc.close()
    return pages_data


def ocr_page(pdf_path, page_num):
    """
    Converts a single PDF page to an image and runs Tesseract OCR on it.
    """
    images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
    if images:
        return pytesseract.image_to_string(images[0])
    return ""


if __name__ == "__main__":
    # Quick test: run this file directly to check ingestion works
    test_folder = "data/raw_pdfs"
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(test_folder, filename)
            print(f"\n--- Processing {filename} ---")
            pages = extract_text_from_pdf(path)
            print(f"Extracted {len(pages)} pages")
            print(f"First page preview: {pages[0]['text'][:200]}")