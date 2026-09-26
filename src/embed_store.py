# embed_store.py
# Purpose: Convert text chunks into vectors (embeddings) and store them in Qdrant,
# plus manage document versioning (via content hashing) and search analytics logging.

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid
import hashlib
from collections import Counter

model = SentenceTransformer("models/all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "policy_docs"
SEARCH_LOG_COLLECTION = "search_logs"


# ----------------------------
# Document storage & versioning
# ----------------------------

def setup_collection():
    """Creates the main document collection in Qdrant if it doesn't already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")


def delete_existing_file_chunks(filename):
    """
    Deletes any chunks already stored for this exact filename.
    Used when REPLACING a file's old version with a new one.
    """
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        )
    )


def get_file_hash(file_bytes):
    """
    Generates a fingerprint of a file's exact content.
    Two files with the same hash have byte-for-byte identical content.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def get_stored_file_hash(filename):
    """
    Looks up the content hash we stored for a given filename, if we have one.
    Returns None if this filename has never been processed before.
    """
    all_points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="filename", match=MatchValue(value=filename))]),
        limit=1,
        with_payload=True
    )
    if all_points:
        return all_points[0].payload.get("file_hash")
    return None


def store_chunks(chunks, file_hash=None):
    """
    Converts each chunk's text into a vector and stores it in Qdrant,
    tagging it with the file's content hash so future uploads can detect
    whether a re-uploaded file is identical, changed, or brand new.
    """
    points = []
    for chunk in chunks:
        vector = model.encode(chunk["text"]).tolist()
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk_position": chunk["chunk_position"],
                "text": chunk["text"],
                "file_hash": file_hash
            }
        )
        points.append(point)

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Stored {len(points)} chunks in Qdrant.")


# ----------------------------
# Search analytics
# ----------------------------

def setup_search_log_collection():
    """Creates a small separate collection just for logging search queries."""
    existing = [c.name for c in client.get_collections().collections]
    if SEARCH_LOG_COLLECTION not in existing:
        client.create_collection(
            collection_name=SEARCH_LOG_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE)  # dummy vector, unused for search
        )


def log_search_query(query):
    """Records a search query so we can later show 'most searched' analytics."""
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=[0.0],
        payload={"query": query.strip().lower()}
    )
    client.upsert(collection_name=SEARCH_LOG_COLLECTION, points=[point])


def get_top_searches(limit=5):
    """Returns the most frequently searched queries, most common first."""
    all_points, _ = client.scroll(collection_name=SEARCH_LOG_COLLECTION, limit=10000, with_payload=True)
    counts = Counter(p.payload["query"] for p in all_points)
    return counts.most_common(limit)


# ----------------------------
# Standalone batch ingestion (unchanged)
# ----------------------------

if __name__ == "__main__":
    import os
    from ingest import extract_text_from_pdf
    from chunk import process_pages_into_chunks

    setup_collection()

    test_folder = "data/raw_pdfs"
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(test_folder, filename)
            print(f"\n=== Processing {filename} ===")

            with open(path, "rb") as f:
                file_hash = get_file_hash(f.read())

            pages = extract_text_from_pdf(path)
            chunks = process_pages_into_chunks(pages, filename)
            store_chunks(chunks, file_hash=file_hash)