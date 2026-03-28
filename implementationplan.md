You want a detailed, implementation-ready plan for a local, CPU-only RAG support chatbot, so I’ll first load the planning skill guidance and then produce a concrete architecture and setup blueprint tailored to your RAM and Windows constraints.

Read skill plan

1. Architecture overview
Recommended architecture (local, CPU-only)
UI layer
Minimal HTML chat page served by FastAPI (lowest RAM overhead among Streamlit/Gradio/minimal HTML options).

API layer
FastAPI endpoints:

POST /chat for user questions
POST /ingest for URL/HTML ingestion trigger
GET /health for service checks
Ingestion layer
Loads website/admin docs from:
URL crawl (same domain allowlist)
Local HTML files
Then:

Clean and normalize text
Chunk text
Create embeddings
Store vectors + metadata
Retrieval layer
HNSWLIB index for nearest-neighbor search.
Metadata stored in lightweight SQLite (or JSONL if you want zero DB dependency).

Generation layer
Ollama local inference for answer generation using retrieved chunks as context.

Guardrails layer
Off-topic filtering and strict prompt policy:

Similarity threshold gate
Refusal behavior when context is missing/off-topic
Optional domain-keyword check for extra safety
Data flow
User question -> question embedding -> vector retrieval top-k -> relevance filter -> prompt assembly (system + retrieved context + user question) -> Ollama generation -> response with cited source snippets.

2. Model + embedding selection for 8GB RAM / no GPU
Chosen LLM
llama3.2:3b (Ollama)

Why this choice
8GB machine with only ~5GB available to app is tight.
8B q4 models often consume too much memory once runtime overhead + context cache + Python process are included.
3B gives better stability, fewer OOM events, and faster CPU latency on i5 13th gen.
For support-chat use case with strong RAG grounding, 3B is usually sufficient.
Chosen embedding model
nomic-embed-text (via Ollama)

Why this embedding choice
Keeps all model serving in one runtime (Ollama), simplifying deployment on Windows.
Avoids adding PyTorch-heavy local embedding stack.
Good quality/size tradeoff for semantic retrieval in admin/support docs.
Practical fallback
If you later confirm stable free RAM above ~6GB for app runtime, test llama3.1:8b-q4 as an optional quality mode. Default should remain llama3.2:3b for reliability.

3. Document ingestion pipeline
Input sources
URL mode:
Crawl only allowed domain/path(s) for the target admin panel.
Skip logout/profile/user-private pages unless explicitly needed.
Respect robots/terms where applicable.
Local file mode:
Load provided HTML files directly from folder.
Extraction and cleaning
Libraries:

requests for fetch
beautifulsoup4 + lxml for parsing
trafilatura (or readability-like cleanup) for noisy page cleanup
Process:

Strip script/style/nav/footer boilerplate
Preserve headings, form labels, table text, button text, error/help text
Normalize whitespace
Keep page title and URL/path metadata
Chunking strategy
Chunk size: 450-700 tokens equivalent (or ~1200-1800 characters)
Overlap: 80-120 tokens (or ~250-350 characters)
Split by semantic boundaries first: headings, paragraphs, list items, table rows
Attach metadata per chunk:
source URL or file
section heading
chunk id
ingestion timestamp
Embedding and indexing
Generate embeddings via Ollama embedding endpoint using nomic-embed-text
Build HNSWLIB index:
Space: cosine
M: 16
efConstruction: 100
Persist:
hnsw index file on disk
metadata map in SQLite/JSONL keyed by vector id
Re-ingestion policy
Support full rebuild and incremental update
Hash chunk text; skip unchanged chunks
Keep simple versioning for rollback
4. RAG query pipeline
Receive user question.
Normalize text (trim, lowercase copy for checks, keep original for answer).
Embed question with nomic-embed-text.
Retrieve top-k (start with k=5) from HNSW index.
Apply relevance gate:
If top score below threshold, refuse as off-topic/insufficient context.
Build context pack:
Keep top 3-5 chunks
Deduplicate near-identical chunks
Cap total context length for low RAM
Compose prompt:
strict system instruction
context snippets with source labels
user question
Generate answer via Ollama llama3.2:3b with low-RAM inference settings.
Return:
concise answer
source references (URL/file + section)
refusal message when no support evidence is found
5. System prompt design (strict domain restriction)
Prompt goals
Answer only from provided admin-panel knowledge base
Refuse off-topic queries
Never fabricate features/settings not present in retrieved context
Prefer short procedural support answers
Required prompt rules
You are an admin support assistant for this specific product/admin interface only.
Use only provided context snippets.
If context is missing, ambiguous, or unrelated, respond with a refusal and ask user to rephrase within admin scope.
Do not answer general knowledge, coding trivia, politics, health, finance, or unrelated domains.
Cite which source section/page the answer came from.
If multiple steps are needed, provide numbered steps.
Off-topic refusal pattern
“I can only help with questions about the provided admin interface documentation. Please ask about features, settings, workflows, or errors from that admin panel.”
Extra guardrail
Use dual checks before generation:

retrieval score threshold
admin-domain keyword/intent filter
If either fails, return refusal directly without calling generation model.
6. File/folder structure
Proposed project structure:

d:/chatbot

app
api
routes_chat.py
routes_ingest.py
routes_health.py
core
config.py
logging.py
prompts.py
guardrails.py
rag
loader.py
cleaner.py
chunker.py
embedder_ollama.py
vector_index_hnsw.py
retriever.py
pipeline.py
ui
templates
chat.html
static
styles.css
app.js
models
schemas.py
main.py
data
raw
processed
index
hnsw.index
metadata.sqlite
scripts
ingest_url.ps1
ingest_local.ps1
rebuild_index.ps1
tests
test_retrieval.py
test_guardrails.py
test_api_chat.py
requirements.txt
.env.example
README.md
7. Step-by-step setup instructions (Windows)
A. Install Ollama
Install via winget:

winget install Ollama.Ollama

Start Ollama service (if not auto-started):

ollama serve

Pull required models:

ollama pull llama3.2:3b
ollama pull nomic-embed-text

B. Create Python environment
From d:/chatbot:

py -3.11 -m venv .venv

Activate:

.venv\Scripts\Activate.ps1

Upgrade pip:

python -m pip install --upgrade pip

C. Install dependencies
Recommended baseline packages:

fastapi
uvicorn[standard]
requests
beautifulsoup4
lxml
trafilatura
hnswlib
numpy
pydantic
python-dotenv
jinja2
httpx
orjson
Install command:

D. Configure environment variables
Use a .env file with:

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_CHAT_MODEL=llama3.2:3b
OLLAMA_EMBED_MODEL=nomic-embed-text
INDEX_PATH=./data/index/hnsw.index
METADATA_DB_PATH=./data/index/metadata.sqlite
RETRIEVAL_TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=(tune after evaluation)
E. Run services
Ensure Ollama running.

Start FastAPI app:

uvicorn app.main:app --host 127.0.0.1 --port 8000

Open local chat UI in browser.

F. Ingest content
Run URL ingestion for admin docs.
Verify index size and metadata count.
Run smoke queries for known admin tasks.
8. Performance tips for low-RAM CPU-only usage
Keep LLM at llama3.2:3b for default runtime.
Limit generation context window (for example 2048-3072 tokens).
Keep retrieval top-k small (3 to 5).
Use short chunk sizes; avoid oversized context packing.
Disable unnecessary logging and verbose tracing.
Avoid heavy rerankers/cross-encoders on CPU in first version.
Ingestion should be batch/offline, not on every chat request.
Pre-clean HTML aggressively to reduce stored tokens.
Keep one active model at a time in Ollama to reduce memory pressure.
Return concise answers to reduce output-token latency.
Cache embeddings for repeated chunks/questions when practical.
Prefer incremental re-indexing over full rebuilds.
9. Risks and limitations
CPU latency
Responses may be slow under concurrent requests; expect single-user or low-concurrency usage.

Hallucination risk
Even with RAG, model can invent details if guardrails are weak. Mitigation: strict prompt + score threshold + refusal policy.

Incomplete scraping
Dynamic JS-rendered admin content may not be captured by simple HTML fetch. Mitigation: optional browser-based crawler stage if needed.

Content drift
Admin UI changes can stale the index. Mitigation: scheduled re-ingestion and change detection.

Memory spikes
Large contexts or accidental 8B usage can trigger slowdowns/OOM. Mitigation: enforce model/ctx caps in config.

Off-topic leakage
If relevance threshold too low, bot may attempt unrelated answers. Mitigation: tune threshold using test question set.

Security/privacy
If docs contain sensitive internal details, local storage must be protected (filesystem permissions, local-only binding, no external exposure).

Recommended implementation baseline (decision summary)
LLM: llama3.2:3b
Embeddings: nomic-embed-text via Ollama
Vector store: HNSWLIB + SQLite metadata
UI: minimal HTML served by FastAPI
Guardrails: retrieval threshold + strict refusal prompt + domain intent filter
If you want, next step can be Section 1 implementation first (skeleton architecture and interfaces only), then we proceed section-by-section exactly as requested.
