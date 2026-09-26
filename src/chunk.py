# chunk.py
# Purpose: Take the page-by-page text from ingest.py, clean obvious OCR junk,
# and split it into smaller overlapping "chunks" that are easier to search and embed.

import re


def clean_ocr_text(text):
    """
    Light cleanup of OCR output: just collapses extra whitespace/newlines.
    Deliberately conservative — we don't try to guess and remove garbled
    words, since that risks deleting real (short) content by mistake.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits a block of text into chunks of roughly `chunk_size` words,
    with `overlap` words repeated between consecutive chunks.

    Why overlap? If a sentence gets cut in half right at a chunk boundary,
    overlap ensures that sentence still appears whole in at least one chunk,
    so we don't lose meaning at the edges.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap

    return chunks


def process_pages_into_chunks(pages_data, filename):
    """
    Takes the list of {"page": N, "text": "..."} dicts from ingest.py,
    cleans each page's text, then splits it into chunk records, each tagged with:
    - which file it came from
    - which page it came from
    - its position among chunks on that page (for citations later)
    """
    all_chunks = []

    for page_info in pages_data:
        page_num = page_info["page"]
        page_text = clean_ocr_text(page_info["text"])  # clean FIRST, then chunk

        text_chunks = chunk_text(page_text)

        for position, chunk in enumerate(text_chunks):
            all_chunks.append({
                "filename": filename,
                "page": page_num,
                "chunk_position": position,
                "text": chunk
            })

    return all_chunks


if __name__ == "__main__":
    import os
    from ingest import extract_text_from_pdf

    test_folder = "data/raw_pdfs"
    sample_file = os.listdir(test_folder)[0]
    sample_path = os.path.join(test_folder, sample_file)

    pages = extract_text_from_pdf(sample_path)
    chunks = process_pages_into_chunks(pages, sample_file)

    print(f"File: {sample_file}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Example chunk record:\n{chunks[0]}")