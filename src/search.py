# search.py
# Purpose: Take a user's search query, find the top 5 most relevant chunks
# in Qdrant, and generate a summary using Azure OpenAI based on those chunks.

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Load the .env file so we can read the Azure credentials without hardcoding them in code
load_dotenv()

# Load the SAME embedding model we used for storing chunks.
# This is important: the query and the stored chunks MUST be embedded
# by the same model, otherwise their vectors aren't comparable at all.
model = SentenceTransformer("models/all-MiniLM-L6-v2")

client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "policy_docs"

# Set up the Azure OpenAI client using the credentials from .env
azure_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


def semantic_search(query, top_k=5):
    """
    Converts the user's query into a vector, then asks Qdrant for the
    'top_k' stored chunks whose vectors are closest in meaning.
    Returns a list of results, each with the matched text + citation info.
    """
    query_vector = model.encode(query).tolist()

    # Newer qdrant-client versions use query_points() instead of search()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )

    matches = []
    for r in response.points:  # results are now inside response.points
        matches.append({
            "score": r.score,
            "filename": r.payload["filename"],
            "page": r.payload["page"],
            "chunk_position": r.payload["chunk_position"],
            "text": r.payload["text"]
        })
    return matches

def generate_summary(query, matches):
    """
    Sends the top matched chunks to Azure OpenAI and asks it to answer
    the user's question using ONLY that content — so the summary stays
    grounded in your actual policy documents, not the model's general knowledge.
    """
    context = "\n\n".join(
        f"[Source: {m['filename']}, page {m['page']}]\n{m['text']}"
        for m in matches
    )

    prompt = f"""You are answering a question using ONLY the provided policy document excerpts below.
If the answer isn't in the excerpts, say so clearly — do not make anything up.

Excerpts:
{context}

Question: {query}

Answer:"""

    response = azure_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    # Quick manual test: type a question, see the matches + generated summary
    test_query = input("Enter a search query: ")

    matches = semantic_search(test_query)

    print("\n--- Top Matches ---")
    for i, m in enumerate(matches, 1):
        print(f"\n{i}. [{m['filename']} - Page {m['page']}] (score: {m['score']:.3f})")
        print(m['text'][:200] + "...")

    print("\n--- Summary ---")
    summary = generate_summary(test_query, matches)
    print(summary)