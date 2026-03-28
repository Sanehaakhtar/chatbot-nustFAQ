import json
import difflib
import sqlite3
from pathlib import Path
import re
from typing import Dict, Generator, List, Optional, Tuple

import hnswlib
import requests


# Configurable constants
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text"
INDEX_PATH = Path("data/index/hnsw.index")
METADATA_DB_PATH = Path("data/index/metadata.sqlite")
SYSTEM_PROMPT_PATH = Path("system_prompt.txt")
TOP_K = 8
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
NUM_CTX = 1536
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 220
MAX_CONTEXT_CHUNKS = 4
MAX_CHUNK_CHARS = 700
SIMILARITY_THRESHOLD = 0.35
REQUEST_TIMEOUT_SECONDS = 240
REFUSAL_MESSAGE = "I can only help with NUST administration topics."
FAQ_MATCH_THRESHOLD = 0.62
FAQ_DIRECT_MATCH_THRESHOLD = 0.80
GREETING_MESSAGE = (
    "Hello! I can help with NUST admissions, fees, test schedules, hostels, and student services. "
    "Try asking: 'What is the eligibility criteria for UG admissions?'"
)

DOMAIN_TERMS = {
    "nust",
    "admin",
    "administration",
    "student",
    "students",
    "enrollment",
    "enrolment",
    "admission",
    "fee",
    "fees",
    "challan",
    "course",
    "courses",
    "registration",
    "register",
    "semester",
    "exam",
    "exams",
    "schedule",
    "hostel",
    "services",
    "portal",
    "attendance",
    "transcript",
    "result",
    "results",
    "gpa",
    "cgpa",
}

GREETING_TERMS = {
    "hi",
    "hello",
    "hey",
    "salam",
    "assalamualaikum",
    "start",
    "help",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

TOKEN_ALIASES = {
    "announced": "announce",
    "announcement": "announce",
    "announcements": "announce",
    "timeline": "schedule",
    "timelines": "schedule",
    "result": "result",
    "results": "result",
    "documents": "document",
    "processing": "process",
    "payments": "payment",
    "fees": "fee",
    "foreigner": "foreign",
    "foreigners": "foreign",
    "international": "foreign",
    "apply": "admission",
    "eligible": "eligibility",
    "eligibility": "eligibility",
    "admissions": "admission",
    "kab": "when",
    "khulenge": "open",
    "khule": "open",
    "honge": "when",
    "kitni": "amount",
    "kitna": "amount",
    "kaise": "how",
    "kese": "how",
    "karna": "process",
    "karen": "process",
    "krna": "process",
    "apply": "admission",
    "fees": "fee",
}


def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        "You are a NUST admin assistant. Only answer questions about NUST administration, "
        "student services, and FAQs. If the question is unrelated, respond: "
        "'I can only help with NUST administration topics.'"
    )


def is_ollama_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def has_domain_terms(question: str) -> bool:
    lower = question.lower()
    return any(term in lower for term in DOMAIN_TERMS)


def is_greeting_or_help(question: str) -> bool:
    cleaned = question.strip().lower()
    if not cleaned:
        return False
    return cleaned in GREETING_TERMS


def tokenize_query_terms(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
    normalized: set[str] = set()
    for token in tokens:
        if token in STOPWORDS:
            continue

        token = TOKEN_ALIASES.get(token, token)
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]

        token = TOKEN_ALIASES.get(token, token)
        if token and token not in STOPWORDS:
            normalized.add(token)

    return normalized


def rerank_chunks(question: str, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    query_terms = tokenize_query_terms(question)

    def score(chunk: Dict[str, str]) -> float:
        base = float(chunk.get("similarity", "0") or 0.0)
        content = f"{chunk.get('title', '')} {chunk.get('content', '')}".lower()
        overlap = sum(1 for term in query_terms if term in content)
        source = chunk.get("source", "").lower()
        source_boost = 0.10 if "nust_admin.html" in source else 0.0
        return base + (0.03 * overlap) + source_boost

    return sorted(chunks, key=score, reverse=True)


def parse_faq_qa(content: str) -> Optional[Tuple[str, str]]:
    match = re.search(r"Question:\s*(.+?)\s*Answer:\s*(.+)", content, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    question = match.group(1).strip()
    answer = match.group(2).strip()
    if not question or not answer:
        return None
    return question, answer


def find_fast_faq_answer(question: str, chunks: List[Dict[str, str]]) -> Optional[Tuple[str, List[str]]]:
    q_norm = question.strip().lower()
    query_terms = tokenize_query_terms(question)
    best_score = 0.0
    best_answer: Optional[str] = None
    best_source: Optional[str] = None

    for chunk in chunks:
        parsed = parse_faq_qa(chunk.get("content", ""))
        if not parsed:
            continue
        faq_q, faq_a = parsed
        ratio = difflib.SequenceMatcher(None, q_norm, faq_q.lower()).ratio()
        faq_terms = tokenize_query_terms(faq_q)
        overlap = len(query_terms & faq_terms)
        overlap_ratio = overlap / max(1, len(query_terms))
        score = (0.75 * ratio) + (0.25 * overlap_ratio)
        if score > best_score:
            best_score = score
            best_answer = faq_a
            best_source = f"{chunk['title']} ({chunk['section']}) - {chunk['source']}"

    if best_answer and best_source and best_score >= FAQ_MATCH_THRESHOLD:
        return best_answer, [best_source]
    return None


def embed_text(text: str) -> List[float]:
    payload = {"model": EMBED_MODEL, "prompt": text}
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("Embedding response was empty.")
    return embedding


def open_metadata_db() -> sqlite3.Connection:
    conn = sqlite3.connect(METADATA_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def total_chunks(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
    return int(row["count"]) if row else 0


def load_index(embedding_dim: int, max_elements: int) -> hnswlib.Index:
    index = hnswlib.Index(space="cosine", dim=embedding_dim)
    index.load_index(str(INDEX_PATH), max_elements=max(10000, max_elements + 1000))
    index.set_ef(50)
    return index


def fetch_chunk_by_id(conn: sqlite3.Connection, chunk_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, source, title, section, content
        FROM chunks
        WHERE id = ?
        """,
        (chunk_id,),
    ).fetchone()


def retrieve_context(question: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    if not INDEX_PATH.exists() or not METADATA_DB_PATH.exists():
        return [], "Vector store is empty. Run ingestion first with ingest.py."

    conn = open_metadata_db()
    try:
        count = total_chunks(conn)
        if count == 0:
            return [], "Vector store is empty. Run ingestion first with ingest.py."

        query_embedding = embed_text(question)
        index = load_index(len(query_embedding), count)
        k = min(TOP_K, count)
        labels, distances = index.knn_query([query_embedding], k=k)

        chunks: List[Dict[str, str]] = []
        for label, distance in zip(labels[0], distances[0]):
            similarity = 1.0 - float(distance)
            if similarity < SIMILARITY_THRESHOLD:
                continue

            row = fetch_chunk_by_id(conn, int(label))
            if row is None:
                continue

            chunks.append(
                {
                    "id": str(row["id"]),
                    "source": str(row["source"]),
                    "title": str(row["title"]),
                    "section": str(row["section"]),
                    "content": str(row["content"]),
                    "similarity": f"{similarity:.3f}",
                }
            )

        ranked_chunks = rerank_chunks(question, chunks)
        return ranked_chunks[:MAX_CONTEXT_CHUNKS], None
    except requests.RequestException:
        return [], "Could not reach Ollama. Start it with: ollama serve"
    except Exception as exc:
        return [], f"Retrieval failed: {exc}"
    finally:
        conn.close()


def build_context_block(chunks: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        content = chunk["content"]
        if len(content) > MAX_CHUNK_CHARS:
            content = content[:MAX_CHUNK_CHARS].rstrip() + " ..."
        lines.append(
            f"[{i}] Source: {chunk['source']} | Title: {chunk['title']} | Section: {chunk['section']}\n"
            f"{content}"
        )
    return "\n\n".join(lines)


def build_user_prompt(question: str, context_block: str) -> str:
    return (
        "Answer the user using only the context below. "
        "You may combine multiple context snippets to infer a concise answer when the exact wording is not present. "
        "If context is still insufficient, refuse with the required refusal message.\n\n"
        f"Context:\n{context_block}\n\n"
        f"User question: {question}\n\n"
        "Response requirements:\n"
        "1) Keep answers concise and practical for student/admin support.\n"
        "2) If steps are needed, use numbered steps.\n"
        "3) Mention source title or section when possible."
    )


def stream_generate(prompt: str, system_prompt: str) -> Generator[str, None, None]:
    payload = {
        "model": CHAT_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": TEMPERATURE,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    }

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            token = data.get("response", "")
            if token:
                yield token
            if data.get("done", False):
                break


def format_sources(chunks: List[Dict[str, str]]) -> List[str]:
    sources: List[str] = []
    for chunk in chunks:
        sources.append(
            f"{chunk['title']} ({chunk['section']}) - {chunk['source']}"
        )
    return sources


class NustRAG:
    def __init__(self) -> None:
        self.system_prompt = load_system_prompt()
        self.faq_cache = self._load_faq_cache()
        self.faq_term_vocab = self._build_faq_term_vocab()

    def _load_faq_cache(self) -> List[Dict[str, str]]:
        if not METADATA_DB_PATH.exists():
            return []

        cache: List[Dict[str, str]] = []
        conn = open_metadata_db()
        try:
            rows = conn.execute(
                """
                SELECT source, title, section, content
                FROM chunks
                WHERE section LIKE 'faq_%'
                """
            ).fetchall()
            for row in rows:
                parsed = parse_faq_qa(str(row["content"]))
                if not parsed:
                    continue
                question, answer = parsed
                cache.append(
                    {
                        "question": question,
                        "answer": answer,
                        "source": str(row["source"]),
                        "title": str(row["title"]),
                        "section": str(row["section"]),
                    }
                )
        finally:
            conn.close()

        return cache

    def _build_faq_term_vocab(self) -> set[str]:
        vocab: set[str] = set()
        for item in self.faq_cache:
            vocab.update(tokenize_query_terms(item["question"]))
        return vocab

    def _should_refuse_early(self, question: str) -> bool:
        query_terms = tokenize_query_terms(question)
        if not query_terms:
            return False

        overlap = len(query_terms & self.faq_term_vocab)
        return overlap == 0 and not has_domain_terms(question)

    def _find_direct_faq_answer(self, question: str) -> Optional[Tuple[str, List[str]]]:
        if not self.faq_cache:
            return None

        q_norm = question.strip().lower()
        query_terms = tokenize_query_terms(question)
        best_score = 0.0
        best_item: Optional[Dict[str, str]] = None

        for item in self.faq_cache:
            faq_q = item["question"].lower()
            ratio = difflib.SequenceMatcher(None, q_norm, faq_q).ratio()
            faq_terms = tokenize_query_terms(item["question"])
            overlap = len(query_terms & faq_terms)
            overlap_ratio = overlap / max(1, len(query_terms))
            score = (0.40 * ratio) + (0.60 * overlap_ratio)

            # Favor paraphrases that preserve all key terms.
            if overlap_ratio >= 0.99 and ratio >= 0.45:
                score += 0.05

            # Handle short intent-style prompts like "can foreigner apply".
            if len(query_terms) <= 4 and overlap_ratio >= 0.66 and ratio >= 0.28:
                score += 0.22

            if score > best_score:
                best_score = score
                best_item = item

        if not best_item or best_score < FAQ_DIRECT_MATCH_THRESHOLD:
            return None

        source = f"{best_item['title']} ({best_item['section']}) - {best_item['source']}"
        return best_item["answer"], [source]

    def precheck(self, question: str) -> Optional[str]:
        if not question.strip():
            return "Please enter a question."
        if not is_ollama_available():
            return "Ollama is not running. Start it with: ollama serve"
        if is_greeting_or_help(question):
            return GREETING_MESSAGE
        return None

    def answer(self, question: str) -> Dict[str, object]:
        precheck_error = self.precheck(question)
        if precheck_error:
            return {
                "answer": precheck_error,
                "sources": [],
                "status": "blocked",
            }

        direct_faq = self._find_direct_faq_answer(question)
        if direct_faq:
            answer, sources = direct_faq
            return {
                "answer": answer,
                "sources": sources,
                "status": "ok",
            }

        if self._should_refuse_early(question):
            return {
                "answer": REFUSAL_MESSAGE,
                "sources": [],
                "status": "blocked",
            }

        chunks, retrieval_error = retrieve_context(question)
        if retrieval_error:
            return {
                "answer": retrieval_error,
                "sources": [],
                "status": "error",
            }

        if not chunks:
            return {
                "answer": REFUSAL_MESSAGE,
                "sources": [],
                "status": "blocked",
            }

        fast_faq = find_fast_faq_answer(question, chunks)
        if fast_faq:
            answer, sources = fast_faq
            return {
                "answer": answer,
                "sources": sources,
                "status": "ok",
            }

        prompt = build_user_prompt(question, build_context_block(chunks))
        try:
            tokens = []
            for token in stream_generate(prompt, self.system_prompt):
                tokens.append(token)
        except requests.RequestException:
            return {
                "answer": "Could not generate response. Check Ollama and model availability.",
                "sources": [],
                "status": "error",
            }
        except Exception as exc:
            return {
                "answer": f"Generation failed: {exc}",
                "sources": [],
                "status": "error",
            }

        return {
            "answer": "".join(tokens).strip() or REFUSAL_MESSAGE,
            "sources": format_sources(chunks),
            "status": "ok",
        }

    def stream_answer(self, question: str) -> Generator[str, None, None]:
        precheck_error = self.precheck(question)
        if precheck_error:
            yield precheck_error
            return

        direct_faq = self._find_direct_faq_answer(question)
        if direct_faq:
            answer, sources = direct_faq
            yield answer
            if sources:
                yield "\n\nSources:\n- " + "\n- ".join(sources)
            return

        if self._should_refuse_early(question):
            yield REFUSAL_MESSAGE
            return

        chunks, retrieval_error = retrieve_context(question)
        if retrieval_error:
            yield retrieval_error
            return

        if not chunks:
            yield REFUSAL_MESSAGE
            return

        fast_faq = find_fast_faq_answer(question, chunks)
        if fast_faq:
            answer, sources = fast_faq
            yield answer
            if sources:
                yield "\n\nSources:\n- " + "\n- ".join(sources)
            return

        prompt = build_user_prompt(question, build_context_block(chunks))
        try:
            for token in stream_generate(prompt, self.system_prompt):
                yield token
        except requests.RequestException:
            yield "Could not generate response. Check Ollama and model availability."
            return
        except Exception as exc:
            yield f"Generation failed: {exc}"
            return

        sources = format_sources(chunks)
        if sources:
            yield "\n\nSources:\n- " + "\n- ".join(sources)
