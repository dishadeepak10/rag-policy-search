# chunk.py
# Purpose: Take the page-by-page text from ingest.py, clean obvious OCR junk,
# and split it into smaller overlapping "chunks" — now chunking across the
# WHOLE document (not page-by-page), so paragraphs split across a page
# boundary stay together as one meaningful piece, as requested.

import re


def clean_ocr_text(text):
    """
    Light cleanup of OCR output: just collapses extra whitespace/newlines.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def process_pages_into_chunks(pages_data, filename, chunk_size=500, overlap=50):
    """
    Combines ALL pages into one continuous stream of words first, while
    remembering which page each word came from. Then chunks across that
    full stream — so a paragraph split between, say, Page 5 and Page 6
    ends up together in the same chunk, instead of being cut in half.
    """
    # Step 1: flatten every page's words into one list, tagged with their page number
    all_words_with_pages = []
    for page_info in pages_data:
        page_num = page_info["page"]
        cleaned = clean_ocr_text(page_info["text"])
        for word in cleaned.split():
            all_words_with_pages.append((word, page_num))

    # Step 2: slide a chunking window across this COMBINED list
    all_chunks = []
    start = 0
    position = 0

    while start < len(all_words_with_pages):
        end = start + chunk_size
        chunk_slice = all_words_with_pages[start:end]

        chunk_words = [w for w, p in chunk_slice]
        chunk_text_str = " ".join(chunk_words)

        # A chunk might now span more than one page — record that as a range
        pages_in_chunk = sorted(set(p for w, p in chunk_slice))
        page_label = pages_in_chunk[0] if len(pages_in_chunk) == 1 else f"{pages_in_chunk[0]}-{pages_in_chunk[-1]}"

        all_chunks.append({
            "filename": filename,
            "page": page_label,
            "chunk_position": position,
            "text": chunk_text_str
        })

        position += 1
        start += chunk_size - overlap

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