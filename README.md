# NUST Local RAG Chatbot

## Description
This project is a local AI chatbot for NUST administration and student service queries. It uses retrieval-augmented generation with a local LLM runtime and a local vector index.

## Features
- Domain-focused answering for NUST admin topics
- Fast FAQ-style matching for common queries
- Contextual reasoning over retrieved chunks for non-exact prompts
- Basic Roman Urdu intent normalization for admission and fee questions
- Markdown rendering in chatbot responses
- Separate source metadata section under each answer
- Static model evaluation page for demo presentation
- Light and dark mode UI toggle

## Tech Stack
- Python 3.11 or 3.12
- FastAPI
- Ollama
- gemma3:4b
- nomic-embed-text
- hnswlib
- SQLite

## Setup Instructions
1. Install Python 3.11 or 3.12.
2. Install Ollama and start it.

```powershell
winget install Ollama.Ollama
ollama serve
```

3. Pull required models.

```powershell
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

4. Create and activate virtual environment.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Ingest NUST FAQ data.

```powershell
python ingest.py --url "https://nust.edu.pk/faqs/" --url "https://nust.edu.pk/faq-category/ug-admission/" --url "https://nust.edu.pk/faq-category/mbbs-admissions-faqs/" --url "https://nust.edu.pk/faq-category/bshnd-admissions-faqs/"
```

6. Start the app.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_server.ps1 -Port 8001
```

7. Open the app.

```text
http://127.0.0.1:8001
```

## How It Works
1. Data is ingested from NUST FAQ pages and stored in SQLite plus HNSW index.
2. On each query, the system tries fast FAQ matching first.
3. If exact FAQ match is not enough, retrieval is performed from indexed chunks.
4. The model generates a concise answer from retrieved context.
5. Sources are shown separately as metadata.

## Limitations
- The evaluation page is static and demo-only. It does not run real benchmark scoring.
- Output quality depends on ingested data quality and coverage.
- This is designed for NUST admin domain and refuses unrelated queries.

## Author
Saneha Akhtar  
CMS: 517085
