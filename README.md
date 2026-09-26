# Company Policy Document Search (RAG Project)

A semantic search system for company policy PDFs, using OCR, vector embeddings, and Azure OpenAI-generated summaries with citations.

## Tech Stack
- **Streamlit** — web interface
- **Qdrant** — vector database (runs via Docker)
- **Sentence-Transformers (all-MiniLM-L6-v2)** — text embeddings
- **Tesseract + Poppler** — OCR for scanned PDFs
- **Azure OpenAI (gpt-4o-mini)** — summary generation

## Prerequisites
- Python 3.12
- Docker Desktop
- Tesseract OCR installed (path set in `ingest.py`)
- Poppler installed (path set in `ingest.py`)
- Azure OpenAI credentials in `.env`

## Setup
```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Qdrant
docker compose up -d

# 4. Fill in .env with Azure OpenAI credentials

# 5. Ingest documents (one-time, or after adding new PDFs manually to data/raw_pdfs)
python src\embed_store.py

# 6. Run the app
streamlit run src\app.py
```

## Project Structure
rag_project/
├── data/raw_pdfs/ # Source policy PDFs (confidential — not in Git)
├── models/ # Local embedding model files
├── src/
│ ├── ingest.py # PDF text extraction + OCR fallback
│ ├── chunk.py # Text cleaning + chunking
│ ├── embed_store.py # Embedding generation + Qdrant storage
│ ├── search.py # Semantic search + Azure OpenAI summary
│ └── app.py # Streamlit UI
├── docker-compose.yml # Qdrant container config
└── requirements.txt

## Features
- Upload new policy PDFs anytime (auto-replaces old version if re-uploaded)
- Remove individual documents or clear the entire database
- Search returns top 5 relevant excerpts with file/page citations
- AI-generated summary grounded strictly in matched document content

## Known Limitations
- PDF only (per current scope)
- OCR text may contain minor artifacts from scanned page edges on older documents