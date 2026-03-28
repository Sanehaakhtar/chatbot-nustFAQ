import argparse
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import hnswlib
import requests
import trafilatura
from bs4 import BeautifulSoup
import cloudscraper


# Configurable constants
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "nust_admin_docs"
INDEX_PATH = Path("data/index/hnsw.index")
METADATA_DB_PATH = Path("data/index/metadata.sqlite")
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
HNSW_SPACE = "cosine"
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100
HNSW_EF_SEARCH = 50
DEFAULT_MAX_ELEMENTS = 10000
INDEX_GROWTH_BUFFER = 1000
REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "NUST-RAG-Ingest/1.0"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def is_supported_python() -> bool:
    # hnswlib/numpy wheels used by this project are stable on 3.11/3.12 on Windows.
    return sys.version_info[:2] <= (3, 12)


@dataclass
class ChunkRecord:
    source: str
    title: str
    section: str
    text: str


@dataclass
class SourceDoc:
    source: str
    title: str
    text: str
    html: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(METADATA_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            chunk_hash TEXT UNIQUE NOT NULL,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            section TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(chunk_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT OR IGNORE INTO settings(key, value) VALUES ('collection_name', ?)",
        (COLLECTION_NAME,),
    )
    conn.commit()
    return conn


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def html_to_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()

    title = normalize_text(soup.title.get_text(" ", strip=True) if soup.title else "Untitled")
    html_clean = str(soup)

    extracted = trafilatura.extract(
        html_clean,
        include_comments=False,
        include_tables=True,
        include_links=True,
        favor_recall=True,
    )

    if extracted:
        text = normalize_text(extracted)
    else:
        text = normalize_text(soup.get_text(" ", strip=True))

    return title or "Untitled", text


def tokenize_words(text: str) -> List[str]:
    return text.split()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = tokenize_words(text)
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def compute_hash(source: str, text: str) -> str:
    digest = hashlib.sha256()
    digest.update(source.encode("utf-8"))
    digest.update(b"||")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def fetch_url(url: str) -> Optional[SourceDoc]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS, headers=headers)
        response.raise_for_status()
    except requests.RequestException as exc:
        # Retry once with a browser-like user agent for sites that block default clients.
        fallback_headers = dict(headers)
        fallback_headers["User-Agent"] = BROWSER_USER_AGENT
        try:
            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers=fallback_headers,
            )
            response.raise_for_status()
        except requests.RequestException:
            try:
                scraper = cloudscraper.create_scraper()
                response = scraper.get(
                    url,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    headers=fallback_headers,
                )
                response.raise_for_status()
            except requests.RequestException:
                print(f"[WARN] Failed to fetch URL '{url}': {exc}")
                return None

    title, text = html_to_text(response.text)
    if not text:
        print(f"[WARN] No extractable text from URL '{url}'.")
        return None

    return SourceDoc(source=url, title=title, text=text, html=response.text)


def read_html_file(path: Path) -> Optional[SourceDoc]:
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        print(f"[WARN] Failed to read file '{path}': {exc}")
        return None

    title, text = html_to_text(html)
    if not text:
        print(f"[WARN] No extractable text from file '{path}'.")
        return None

    source = str(path.resolve())
    return SourceDoc(source=source, title=title, text=text, html=html)


def gather_sources(urls: Iterable[str], html_dir: Optional[Path]) -> List[SourceDoc]:
    docs: List[SourceDoc] = []

    for url in urls:
        result = fetch_url(url.strip())
        if result:
            docs.append(result)

    if html_dir:
        if not html_dir.exists() or not html_dir.is_dir():
            print(f"[WARN] html-dir does not exist or is not a directory: {html_dir}")
        else:
            for file_path in sorted(html_dir.rglob("*.html")):
                result = read_html_file(file_path)
                if result:
                    docs.append(result)

    return docs


def faq_chunks_from_html(doc: SourceDoc) -> List[ChunkRecord]:
    soup = BeautifulSoup(doc.html, "lxml")
    cards = soup.select("div.card")
    if not cards:
        return []

    seen_questions: set[str] = set()
    records: List[ChunkRecord] = []
    faq_index = 1

    for card in cards:
        q_node = card.select_one("button span")
        a_node = card.select_one("div.card-body")
        if not q_node or not a_node:
            continue

        question = normalize_text(q_node.get_text(" ", strip=True))
        answer = normalize_text(a_node.get_text(" ", strip=True))
        if not question or not answer or not question.endswith("?"):
            continue

        key = question.lower()
        if key in seen_questions:
            continue
        seen_questions.add(key)

        records.append(
            ChunkRecord(
                source=doc.source,
                title=doc.title,
                section=f"faq_{faq_index}",
                text=f"Question: {question}\nAnswer: {answer}",
            )
        )
        faq_index += 1

    return records


def chunks_for_docs(docs: List[SourceDoc]) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    for doc in docs:
        faq_records = faq_chunks_from_html(doc)
        if faq_records:
            records.extend(faq_records)
            continue

        source, title, text = doc.source, doc.title, doc.text
        chunks = chunk_text(text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
        for i, chunk in enumerate(chunks, start=1):
            records.append(
                ChunkRecord(
                    source=source,
                    title=title,
                    section=f"chunk_{i}",
                    text=chunk,
                )
            )
    return records


def ollama_embed(text: str) -> Optional[List[float]]:
    payload = {"model": EMBED_MODEL, "prompt": text}
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    data = response.json()
    emb = data.get("embedding")
    if not isinstance(emb, list) or not emb:
        return None
    return emb


def check_ollama_ready() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def existing_hashes(conn: sqlite3.Connection) -> set:
    rows = conn.execute("SELECT chunk_hash FROM chunks").fetchall()
    return {row[0] for row in rows}


def next_label_start(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()
    return int(row[0]) + 1


def upsert_chunk(
    conn: sqlite3.Connection,
    chunk_hash: str,
    source: str,
    title: str,
    section: str,
    content: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO chunks(chunk_hash, source, title, section, content, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (chunk_hash, source, title, section, content, now_iso()),
    )
    return int(cursor.lastrowid)


def prepare_index(dim: int, needed_total: int) -> hnswlib.Index:
    index = hnswlib.Index(space=HNSW_SPACE, dim=dim)

    if INDEX_PATH.exists():
        max_elements = max(DEFAULT_MAX_ELEMENTS, needed_total + INDEX_GROWTH_BUFFER)
        index.load_index(str(INDEX_PATH), max_elements=max_elements)
        current_max = index.get_max_elements()
        if needed_total > current_max:
            index.resize_index(needed_total + INDEX_GROWTH_BUFFER)
    else:
        max_elements = max(DEFAULT_MAX_ELEMENTS, needed_total + INDEX_GROWTH_BUFFER)
        index.init_index(
            max_elements=max_elements,
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
        )

    index.set_ef(HNSW_EF_SEARCH)
    return index


def ingest(urls: List[str], html_dir: Optional[Path]) -> int:
    ensure_dirs()

    if not check_ollama_ready():
        print("[ERROR] Ollama is not reachable at http://127.0.0.1:11434. Start it with: ollama serve")
        return 1

    docs = gather_sources(urls, html_dir)
    if not docs:
        print("[ERROR] No documents found from the provided URL(s)/HTML directory.")
        return 1

    records = chunks_for_docs(docs)
    if not records:
        print("[ERROR] Document extraction produced zero chunks. Nothing to index.")
        return 1

    conn = open_db()
    known_hashes = existing_hashes(conn)

    new_records: List[ChunkRecord] = []
    new_hashes: List[str] = []

    for record in records:
        h = compute_hash(record.source, record.text)
        if h in known_hashes:
            continue
        known_hashes.add(h)
        new_records.append(record)
        new_hashes.append(h)

    if not new_records:
        print("[INFO] No new chunks to ingest; index is already up to date.")
        conn.close()
        return 0

    print(f"[INFO] New chunks to embed: {len(new_records)}")

    vectors: List[List[float]] = []
    valid_records: List[ChunkRecord] = []

    for i, record in enumerate(new_records, start=1):
        embedding = ollama_embed(record.text)
        if embedding is None:
            print(
                "[ERROR] Embedding request failed. Ensure model is pulled: ollama pull "
                f"{EMBED_MODEL}"
            )
            conn.close()
            return 1
        vectors.append(embedding)
        valid_records.append(record)
        if i % 25 == 0 or i == len(new_records):
            print(f"[INFO] Embedded {i}/{len(new_records)} chunks")

    dim = len(vectors[0])
    existing_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    needed_total = existing_count + len(valid_records)
    index = prepare_index(dim, needed_total)

    labels: List[int] = []
    try:
        for rec, h in zip(valid_records, new_hashes):
            label = upsert_chunk(conn, h, rec.source, rec.title, rec.section, rec.text)
            labels.append(label)

        index.add_items(vectors, labels)
        index.save_index(str(INDEX_PATH))
        conn.commit()
    except Exception as exc:
        conn.rollback()
        print(f"[ERROR] Failed during indexing: {exc}")
        conn.close()
        return 1

    conn.close()
    print(f"[OK] Ingestion complete. Added {len(labels)} chunks to collection '{COLLECTION_NAME}'.")
    print(f"[OK] Index: {INDEX_PATH}")
    print(f"[OK] Metadata DB: {METADATA_DB_PATH}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest NUST admin docs into local HNSWLIB index using Ollama embeddings."
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Admin panel URL to ingest. Use multiple times for multiple URLs.",
    )
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=None,
        help="Directory containing local HTML files to ingest.",
    )
    return parser.parse_args()


def main() -> int:
    if not is_supported_python():
        print(
            "[ERROR] Unsupported Python version detected. "
            "Use Python 3.11 or 3.12 for this project."
        )
        print("[ERROR] Example: py -3.11 -m venv .venv")
        return 1

    args = parse_args()
    urls = [u for u in args.url if u and u.strip()]

    if not urls and args.html_dir is None:
        print("[ERROR] Provide at least one --url or --html-dir.")
        return 1

    return ingest(urls, args.html_dir)


if __name__ == "__main__":
    sys.exit(main())
