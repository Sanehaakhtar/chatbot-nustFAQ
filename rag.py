import json
import difflib
import sqlite3
import time
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, AsyncGenerator

import hnswlib
import httpx


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
EMBED_TIMEOUT_SECONDS = 20
GENERATE_TIMEOUT_SECONDS = 70
SESSION_TTL_SECONDS = 1800
MAX_SESSION_TURNS = 6
REFUSAL_MESSAGE = "I can only help with NUST administration topics."
FAQ_MATCH_THRESHOLD = 0.62
FAQ_DIRECT_MATCH_THRESHOLD = 0.68
FAQ_FALLBACK_THRESHOLD = 0.52
GREETING_MESSAGE = (
    "Hello! I can help with NUST admissions, fees, test schedules, hostels, and student services. "
    "Try asking: 'What is the eligibility criteria for UG admissions?'"
)

FOLLOW_UP_HINTS = {
    "how",
    "how?",
    "details",
    "detail",
    "process",
    "steps",
    "kab",
    "when",
    "kis",
    "kaise",
    "kese",
    "kesy",
    "more",
    "explain",
    "phir",
    "uska",
    "iska",
    "that",
    "this",
}

FOLLOW_UP_PRONOUNS = {
    "how",
    "when",
    "why",
    "details",
    "detail",
    "process",
    "steps",
    "uska",
    "iska",
    "phir",
}

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
    "aoa",
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
    "kya",
    "hai",
    "hain",
    "tha",
    "thi",
    "the",
    "ke",
    "ki",
    "ka",
    "ko",
    "se",
    "me",
    "mein",
    "main",
    "liye",
    "liay",
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
    "registrations": "registration",
    "kab": "when",
    "khulenge": "open",
    "khule": "open",
    "honge": "when",
    "kitni": "amount",
    "kitna": "amount",
    "kaise": "how",
    "kese": "how",
    "kesy": "how",
    "kis": "which",
    "konsi": "which",
    "kon": "which",
    "kitne": "howmany",
    "kitnay": "howmany",
    "zyada": "more",
    "zada": "more",
    "zyaada": "more",
    "program": "programme",
    "programs": "programme",
    "programme": "programme",
    "programmes": "programme",
    "krskte": "can",
    "karsakte": "can",
    "krsakte": "can",
    "kar": "do",
    "sakte": "can",
    "sakta": "can",
    "sakti": "can",
    "applykarsakte": "admission",
    "form": "application",
    "forms": "application",
    "challan": "challan",
    "dastavez": "document",
    "kagzaat": "document",
    "kaagzaat": "document",
    "chahiye": "required",
    "chaahiye": "required",
    "milti": "offer",
    "milta": "offer",
    "milty": "offer",
    "scholarships": "scholarship",
    "financials": "financial",
    "kab": "when",
    "karna": "process",
    "karen": "process",
    "krna": "process",
    "apply": "admission",
    "fees": "fee",
}

ROMAN_URDU_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bek\s+se\s+(zyada|zada|zyaada).*(program|programme|course).*(apply|admission)"), "Can I apply for more than one programme at NUST?"),
    (re.compile(r"\b(registration|admission).*(document|dastavez|kagzaat|kaagzaat)"), "What documents are essentially required to be submitted while applying for admission?"),
    (re.compile(r"\b(document|documents|dastavez|kagzaat|kaagzaat).*(registration|admission|apply)"), "What documents are essentially required to be submitted while applying for admission?"),
    (re.compile(r"\bfee\s+challan.*(status|check|payment)"), "How can I submit the application processing fee (online) using 1Link option?"),
    (re.compile(r"\b(course|semester).*(registration).*(kab|when|open)"), "When does semester course registration open?"),
    (re.compile(r"\b(eligibility|eligible|criteria).*(ug|undergraduate|admission)"), "What is the eligibility criteria for UG admissions?"),
    (re.compile(r"\b(scholarship|financial|assistance).*(milti|milta|available|offer|deti|deta)"), "Does NUST offer scholarship / financial assistance?"),
    (re.compile(r"\b(milti|milta|available|offer|deti|deta).*(scholarship|financial|assistance)"), "Does NUST offer scholarship / financial assistance?"),
]

ENGLISH_INTENT_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b(fee\s+challan|challan|payment\s+status|fee\s+status|processing\s+fee)\b", flags=re.IGNORECASE),
        "How can I submit the application processing fee (online) using 1Link option?",
    ),
]

# Global caches to avoid reloading index/db per request
_INDEX_CACHE: Optional[hnswlib.Index] = None
_INDEX_METADATA_COUNT: int = 0
_HTTP_CLIENT: Optional[httpx.AsyncClient] = None

def get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=GENERATE_TIMEOUT_SECONDS, write=10.0, pool=5.0)
        )
    return _HTTP_CLIENT

def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        "You are a NUST admin assistant. Only answer questions about NUST administration, "
        "student services, and FAQs. If the question is unrelated, respond: "
        "'I can only help with NUST administration topics.'"
    )


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


def rewrite_roman_urdu_question(question: str) -> str:
    normalized = " ".join(re.findall(r"[a-zA-Z0-9]+", question.lower()))
    for pattern, replacement in ROMAN_URDU_PATTERNS:
        if pattern.search(normalized):
            return replacement

    for pattern, replacement in ENGLISH_INTENT_PATTERNS:
        if pattern.search(question):
            return replacement

    return question


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


async def embed_text(text: str) -> List[float]:
    payload = {"model": EMBED_MODEL, "prompt": text}
    client = get_http_client()
    response = await client.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json=payload,
        timeout=httpx.Timeout(connect=5.0, read=EMBED_TIMEOUT_SECONDS, write=10.0, pool=5.0),
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
    global _INDEX_CACHE, _INDEX_METADATA_COUNT
    if _INDEX_CACHE is not None and _INDEX_METADATA_COUNT == max_elements:
        return _INDEX_CACHE
    
    index = hnswlib.Index(space="cosine", dim=embedding_dim)
    index.load_index(str(INDEX_PATH), max_elements=max(10000, max_elements + 1000))
    index.set_ef(50)
    
    _INDEX_CACHE = index
    _INDEX_METADATA_COUNT = max_elements
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


async def retrieve_context(question: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    if not INDEX_PATH.exists() or not METADATA_DB_PATH.exists():
        return [], "Vector store is empty. Run ingestion first with ingest.py."

    conn = open_metadata_db()
    try:
        count = total_chunks(conn)
        if count == 0:
            return [], "Vector store is empty. Run ingestion first with ingest.py."

        query_embedding = await embed_text(question)
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
    except httpx.HTTPError:
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


async def stream_generate(prompt: str, system_prompt: str) -> AsyncGenerator[str, None]:
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

    client = get_http_client()
    async with client.stream(
        "POST",
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
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
        self.faq_exact_lookup = self._build_faq_exact_lookup()
        self.answer_cache: Dict[str, Tuple[float, Dict[str, object]]] = {}
        self.answer_cache_ttl_seconds = 300
        self.sessions: Dict[str, Dict[str, object]] = {}
        self.session_ttl_seconds = SESSION_TTL_SECONDS
        self.max_session_turns = MAX_SESSION_TURNS
        self.stats: Dict[str, int] = {
            "total_requests": 0,
            "cache_hits": 0,
            "direct_hits": 0,
            "fallback_hits": 0,
            "blocked": 0,
        }

    def _prune_sessions(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, payload in self.sessions.items()
            if (now - float(payload.get("updated_at", 0.0))) > self.session_ttl_seconds
        ]
        for sid in expired:
            self.sessions.pop(sid, None)

    def _get_or_create_session(self, session_id: Optional[str]) -> List[Dict[str, str]]:
        if not session_id:
            return []

        self._prune_sessions()
        now = time.time()
        payload = self.sessions.get(session_id)
        if payload is None:
            payload = {"updated_at": now, "turns": []}
            self.sessions[session_id] = payload
        else:
            payload["updated_at"] = now

        turns = payload.get("turns")
        if not isinstance(turns, list):
            payload["turns"] = []
            return payload["turns"]
        return turns

    def _record_turn(self, session_id: Optional[str], user_text: str, assistant_text: str) -> None:
        if not session_id:
            return
        turns = self._get_or_create_session(session_id)

        turns.append({"role": "user", "text": user_text.strip()})
        turns.append({"role": "assistant", "text": assistant_text.strip()})

        max_entries = self.max_session_turns * 2
        if len(turns) > max_entries:
            del turns[: len(turns) - max_entries]

    def _is_follow_up(self, question: str) -> bool:
        tokens = re.findall(r"[a-zA-Z0-9]+", question.lower())
        if not tokens:
            return False

        # Do not rewrite clearly self-contained domain questions.
        if has_domain_terms(question) and len(tokens) > 2:
            return False

        # Very short prompts can be follow-ups.
        if len(tokens) <= 2 and any(token in FOLLOW_UP_HINTS for token in tokens):
            return True

        # Slightly longer prompts must start like a follow-up.
        if len(tokens) <= 4 and tokens[0] in FOLLOW_UP_PRONOUNS:
            return True

        return False

    def _resolve_follow_up(self, question: str, session_id: Optional[str]) -> str:
        turns = self._get_or_create_session(session_id)
        if not turns or not self._is_follow_up(question):
            return question

        last_user = ""
        for turn in reversed(turns):
            if turn.get("role") == "user":
                last_user = str(turn.get("text", "")).strip()
                break

        if not last_user:
            return question

        return f"{last_user}. {question}"

    def _prune_answer_cache(self) -> None:
        now = time.time()
        expired = [
            key
            for key, (ts, _payload) in self.answer_cache.items()
            if (now - ts) > self.answer_cache_ttl_seconds
        ]
        for key in expired:
            self.answer_cache.pop(key, None)

    def get_runtime_metrics(self) -> Dict[str, object]:
        self._prune_sessions()
        self._prune_answer_cache()

        total_requests = self.stats["total_requests"]
        handled = self.stats["direct_hits"] + self.stats["fallback_hits"]
        success_rate = (handled / total_requests) if total_requests else 0.0
        total_session_turns = sum(
            len(payload.get("turns", []))
            for payload in self.sessions.values()
            if isinstance(payload.get("turns"), list)
        )

        return {
            "faq_items": len(self.faq_cache),
            "faq_terms": len(self.faq_term_vocab),
            "requests": total_requests,
            "cache_hits": self.stats["cache_hits"],
            "direct_hits": self.stats["direct_hits"],
            "fallback_hits": self.stats["fallback_hits"],
            "blocked": self.stats["blocked"],
            "success_rate": round(success_rate * 100.0, 1),
            "active_sessions": len(self.sessions),
            "active_turn_entries": total_session_turns,
            "answer_cache_entries": len(self.answer_cache),
        }

    def _cache_key(self, question: str) -> str:
        return " ".join(question.strip().lower().split())

    def _normalize_question(self, question: str) -> str:
        terms = sorted(tokenize_query_terms(question))
        if not terms:
            return self._cache_key(question)
        return " ".join(terms)

    def _build_faq_exact_lookup(self) -> Dict[str, Dict[str, str]]:
        lookup: Dict[str, Dict[str, str]] = {}
        for item in self.faq_cache:
            lookup[self._cache_key(item["question"])] = item
            lookup[self._normalize_question(item["question"])] = item
        return lookup

    def _get_cached_answer(self, question: str) -> Optional[Dict[str, object]]:
        key = self._cache_key(question)
        cached = self.answer_cache.get(key)
        if not cached:
            return None
        ts, payload = cached
        if (time.time() - ts) > self.answer_cache_ttl_seconds:
            self.answer_cache.pop(key, None)
            return None
        return payload

    def _set_cached_answer(self, question: str, payload: Dict[str, object]) -> None:
        key = self._cache_key(question)
        self.answer_cache[key] = (time.time(), payload)

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

    def _format_faq_source(self, item: Dict[str, str]) -> str:
        return f"{item['title']} ({item['section']}) - {item['source']}"

    def _should_refuse_early(self, question: str) -> bool:
        query_terms = tokenize_query_terms(question)
        if not query_terms:
            return False

        overlap = len(query_terms & self.faq_term_vocab)
        return overlap == 0 and not has_domain_terms(question)

    def _find_direct_faq_answer(self, question: str) -> Optional[Tuple[str, List[str]]]:
        if not self.faq_cache:
            return None

        # Fast exact lookup on normalized and raw forms.
        exact_item = self.faq_exact_lookup.get(self._cache_key(question))
        if exact_item:
            return exact_item["answer"], [self._format_faq_source(exact_item)]

        normalized_item = self.faq_exact_lookup.get(self._normalize_question(question))
        if normalized_item:
            return normalized_item["answer"], [self._format_faq_source(normalized_item)]

        q_norm = question.strip().lower()
        query_terms = tokenize_query_terms(question)

        # Deterministic fast path for high-frequency intent.
        if {"document", "registration"}.issubset(query_terms):
            for item in self.faq_cache:
                faq_terms = tokenize_query_terms(item["question"])
                if {"document", "registration"}.issubset(faq_terms):
                    return item["answer"], [self._format_faq_source(item)]

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

            # Favor practical paraphrases with strong term overlap.
            if len(query_terms) >= 3 and overlap_ratio >= 0.60:
                score += 0.12

            # Strong boost for common registration-document intent.
            if {"document", "registration"}.issubset(query_terms) and {"document", "registration"}.issubset(faq_terms):
                score += 0.20

            # Handle short intent-style prompts like "can foreigner apply".
            if len(query_terms) <= 4 and overlap_ratio >= 0.66 and ratio >= 0.28:
                score += 0.22

            if score > best_score:
                best_score = score
                best_item = item

        if not best_item or best_score < FAQ_DIRECT_MATCH_THRESHOLD:
            return None

        return best_item["answer"], [self._format_faq_source(best_item)]

    def _find_faq_fallback_answer(self, question: str) -> Optional[Tuple[str, List[str]]]:
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

            if overlap == 0:
                continue
            if len(query_terms) >= 3 and overlap_ratio < 0.40:
                continue

            score = (0.30 * ratio) + (0.70 * overlap_ratio)
            if overlap_ratio >= 0.5:
                score += 0.08
            if len(query_terms) <= 5 and overlap_ratio >= 0.4:
                score += 0.08

            if score > best_score:
                best_score = score
                best_item = item

        if not best_item or best_score < FAQ_FALLBACK_THRESHOLD:
            return None

        return best_item["answer"], [self._format_faq_source(best_item)]

    def _suggest_faq_questions(self, question: str, limit: int = 3) -> List[str]:
        if not self.faq_cache:
            return []

        q_norm = question.strip().lower()
        query_terms = tokenize_query_terms(question)
        scored: List[Tuple[float, str]] = []
        for item in self.faq_cache:
            faq_question = item["question"]
            faq_terms = tokenize_query_terms(faq_question)
            overlap = len(query_terms & faq_terms)
            if len(query_terms) >= 2 and overlap == 0:
                continue
            overlap_ratio = overlap / max(1, len(query_terms))
            ratio = difflib.SequenceMatcher(None, q_norm, faq_question.lower()).ratio()
            score = (0.25 * ratio) + (0.75 * overlap_ratio)

            if {"admission"} & query_terms and {"admission"} & faq_terms:
                score += 0.10

            scored.append((score, faq_question))

        if not scored:
            # If query terms are too sparse, fall back to lexical similarity only.
            for item in self.faq_cache:
                faq_question = item["question"]
                ratio = difflib.SequenceMatcher(None, q_norm, faq_question.lower()).ratio()
                scored.append((ratio, faq_question))

        scored.sort(key=lambda x: x[0], reverse=True)
        suggestions: List[str] = []
        seen: set[str] = set()
        for _score, faq_question in scored:
            key = faq_question.lower()
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(faq_question)
            if len(suggestions) >= limit:
                break

        return suggestions

    async def precheck(self, question: str) -> Optional[str]:
        if not question.strip():
            return "Please enter a question."
        if is_greeting_or_help(question):
            return GREETING_MESSAGE
        return None

    async def answer(self, question: str, session_id: Optional[str] = None) -> Dict[str, object]:
        self.stats["total_requests"] += 1
        resolved_question = self._resolve_follow_up(question, session_id)
        match_question = rewrite_roman_urdu_question(resolved_question)

        cached = self._get_cached_answer(match_question)
        if cached:
            self.stats["cache_hits"] += 1
            self._record_turn(session_id, question, str(cached.get("answer", "")))
            return cached

        precheck_error = await self.precheck(match_question)
        if precheck_error:
            self.stats["blocked"] += 1
            self._record_turn(session_id, question, precheck_error)
            return {
                "answer": precheck_error,
                "sources": [],
                "status": "blocked",
            }

        direct_faq = self._find_direct_faq_answer(match_question)
        if direct_faq:
            answer, sources = direct_faq
            self.stats["direct_hits"] += 1
            payload = {
                "answer": answer,
                "sources": sources,
                "status": "ok",
            }
            self._set_cached_answer(match_question, payload)
            self._record_turn(session_id, question, answer)
            return payload

        # FAQ-only fast path: use best FAQ fallback before any heavy retrieval/generation.
        fallback_faq = self._find_faq_fallback_answer(match_question)
        if fallback_faq:
            answer, sources = fallback_faq
            self.stats["fallback_hits"] += 1
            payload = {
                "answer": answer,
                "sources": sources,
                "status": "ok",
            }
            self._set_cached_answer(match_question, payload)
            self._record_turn(session_id, question, answer)
            return payload

        if self._should_refuse_early(match_question):
            self.stats["blocked"] += 1
            self._record_turn(session_id, question, REFUSAL_MESSAGE)
            return {
                "answer": REFUSAL_MESSAGE,
                "sources": [],
                "status": "blocked",
            }

        self.stats["blocked"] += 1
        suggestions = self._suggest_faq_questions(match_question, limit=3)
        fallback_message = "I could not find an exact FAQ match."
        if suggestions:
            fallback_message += " Try one of these:\n" + "\n".join(
                [f"{idx}. {q}" for idx, q in enumerate(suggestions, start=1)]
            )
        else:
            fallback_message += " Please rephrase your question using FAQ terms."
        self._record_turn(session_id, question, fallback_message)
        return {
            "answer": fallback_message,
            "sources": [],
            "status": "blocked",
        }

    async def stream_answer(self, question: str, session_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        self.stats["total_requests"] += 1
        resolved_question = self._resolve_follow_up(question, session_id)
        match_question = rewrite_roman_urdu_question(resolved_question)

        precheck_error = await self.precheck(match_question)
        if precheck_error:
            self.stats["blocked"] += 1
            self._record_turn(session_id, question, precheck_error)
            yield precheck_error
            return

        direct_faq = self._find_direct_faq_answer(match_question)
        if direct_faq:
            answer, sources = direct_faq
            self.stats["direct_hits"] += 1
            self._record_turn(session_id, question, answer)
            yield answer
            if sources:
                yield "\n\nSources:\n- " + "\n- ".join(sources)
            return

        fallback_faq = self._find_faq_fallback_answer(match_question)
        if fallback_faq:
            answer, sources = fallback_faq
            self.stats["fallback_hits"] += 1
            self._record_turn(session_id, question, answer)
            yield answer
            if sources:
                yield "\n\nSources:\n- " + "\n- ".join(sources)
            return

        if self._should_refuse_early(match_question):
            self.stats["blocked"] += 1
            self._record_turn(session_id, question, REFUSAL_MESSAGE)
            yield REFUSAL_MESSAGE
            return

        self.stats["blocked"] += 1
        suggestions = self._suggest_faq_questions(match_question, limit=3)
        fallback_message = "I could not find an exact FAQ match."
        if suggestions:
            fallback_message += " Try one of these:\n" + "\n".join(
                [f"{idx}. {q}" for idx, q in enumerate(suggestions, start=1)]
            )
        else:
            fallback_message += " Please rephrase your question using FAQ terms."
        self._record_turn(session_id, question, fallback_message)
        yield fallback_message
