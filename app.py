from typing import Dict, AsyncGenerator, Optional
from uuid import uuid4

from fastapi import FastAPI, Query, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from rag import NustRAG


# Configurable constants
APP_TITLE = "NUST Admin RAG Assistant"
APP_HOST = "127.0.0.1"
APP_PORT = 8001

app = FastAPI(title=APP_TITLE)
rag = NustRAG()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    status: str
    session_id: str


def sse_event(data: str) -> str:
    payload = data.replace("\r", " ").replace("\n", " ")
    return f"data: {payload}\n\n"


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": APP_TITLE}


@app.get("/favicon.ico")
def favicon() -> Response:
    # Return an empty successful response so browsers stop logging favicon 404s.
    return Response(status_code=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_probe() -> Response:
    # Chrome probes this path automatically; return 204 to avoid noisy 404 logs.
    return Response(status_code=204)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> Dict[str, object]:
    session_id = request.session_id or uuid4().hex
    payload = await rag.answer(request.message, session_id=session_id)
    payload["session_id"] = session_id
    return payload


@app.get("/chat/stream")
async def chat_stream(
    message: str = Query(..., min_length=1),
    session_id: Optional[str] = Query(default=None),
) -> StreamingResponse:
    resolved_session_id = session_id or uuid4().hex

    async def generator() -> AsyncGenerator[str, None]:
        yield sse_event(f"[SESSION]{resolved_session_id}")
        yield sse_event("[START]")
        async for token in rag.stream_answer(message, session_id=resolved_session_id):
            yield sse_event(token)
        yield sse_event("[END]")

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/metrics")
def metrics() -> Dict[str, object]:
    return rag.get_runtime_metrics()


@app.get("/evaluation", response_class=HTMLResponse)
def evaluation() -> str:
    metrics = rag.get_runtime_metrics()
    return f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Runtime Evaluation</title>
</head>
<body style=\"font-family:Manrope,sans-serif;background:#f6eff5;padding:20px;color:#1a1730;\">
  <div style=\"max-width:760px;margin:0 auto;background:#fff;border:1px solid #dccedf;border-radius:16px;padding:18px;\">
    <h1 style=\"margin:0 0 6px;\">Runtime Evaluation</h1>
    <p style=\"margin:0 0 14px;color:#6f6480;\">Live metrics from this running server instance.</p>
    <ul style=\"line-height:1.8;\">
      <li>Handled Request Rate: <strong>{metrics['success_rate']}%</strong></li>
      <li>Total Requests: <strong>{metrics['requests']}</strong></li>
      <li>Direct FAQ Hits: <strong>{metrics['direct_hits']}</strong></li>
      <li>Fallback FAQ Hits: <strong>{metrics['fallback_hits']}</strong></li>
      <li>Blocked/Out-of-scope: <strong>{metrics['blocked']}</strong></li>
      <li>Active Sessions: <strong>{metrics['active_sessions']}</strong></li>
      <li>FAQ Entries / Terms: <strong>{metrics['faq_items']} / {metrics['faq_terms']}</strong></li>
      <li>Answer Cache Entries: <strong>{metrics['answer_cache_entries']}</strong></li>
    </ul>
    <div style=\"display:flex;gap:10px;flex-wrap:wrap;\">
      <a href=\"/\" style=\"text-decoration:none;padding:10px 12px;border-radius:10px;background:#982598;color:#fff;\">Back to Chatbot</a>
      <a href=\"/health\" style=\"text-decoration:none;padding:10px 12px;border-radius:10px;border:1px solid #d8c7d5;color:#1a1730;\">Health Check</a>
      <a href=\"/metrics\" style=\"text-decoration:none;padding:10px 12px;border-radius:10px;border:1px solid #d8c7d5;color:#1a1730;\">Raw Metrics JSON</a>
    </div>
  </div>
</body>
</html>
    """


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NUST Admissions Copilot</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #f1e9e9;
      --bg-soft: #f7f1f1;
      --bg-accent: #e491c9;
      --ink: #15173d;
      --muted: #6f6480;
      --panel: #f6f0f4;
      --line: #d8c7d5;
      --brand: #982598;
      --brand-strong: #7f1f7f;
      --bot: #f1e9e9;
      --user: #efdff0;
      --danger: #aa3d3d;
      --shadow: 0 12px 30px rgba(21, 23, 61, 0.12);
      --input-shadow: 0 0 0 3px rgba(152, 37, 152, 0.2);

      --home-bg-base: #fcf6fb;
      --home-glow-a: #ff66c44a;
      --home-glow-b: #ff8fd63b;
      --home-center-glow: #f6b0e45e;
      --home-card-top: rgba(255, 249, 255, 0.98);
      --home-card-bottom: rgba(245, 233, 252, 0.96);
      --home-title: #1a1630;
      --home-sub: #4a3f68;
      --home-hint: #3f2d74;
    }

    body[data-theme="dark"] {
      --bg: #121212;
      --bg-soft: #181818;
      --bg-accent: #0f0f0f;
      --ink: #f3eef7;
      --muted: #bbb2c7;
      --panel: #1b1b1f;
      --line: #2f2f35;
      --brand: #e491c9;
      --brand-strong: #f0b8dd;
      --bot: #202026;
      --user: #35213f;
      --danger: #ffb7c8;
      --shadow: 0 16px 34px rgba(5, 6, 20, 0.38);
      --input-shadow: 0 0 0 3px rgba(228, 145, 201, 0.26);

      --home-bg-base: #0f0f0f;
      --home-glow-a: #ff66c426;
      --home-glow-b: #5170ff26;
      --home-center-glow: #ff66c43b;
      --home-card-top: rgba(24, 24, 28, 0.92);
      --home-card-bottom: rgba(195, 67, 157, 0.94);
      --home-title: #ffffff;
      --home-sub: #c9c6d4;
      --home-hint: #ece4ff;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Manrope", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 8%, #f2d8e8 0%, transparent 34%),
        radial-gradient(circle at 86% 14%, #e6cce8 0%, transparent 30%),
        radial-gradient(circle at 18% 86%, #edd7e7 0%, transparent 22%),
        linear-gradient(180deg, var(--bg-accent) 0%, var(--bg) 62%);
      transition: background 220ms ease, color 220ms ease;
      overflow: hidden;
    }

    .flow {
      width: 100%;
      height: 200vh;
      transform: translateY(0);
      transition: transform 540ms cubic-bezier(0.22, 1, 0.36, 1);
      will-change: transform;
    }

    .flow.chat-open {
      transform: translateY(-100vh);
    }

    .screen {
      min-height: 100vh;
      width: 100%;
      display: grid;
      place-items: center;
      padding: 16px;
      touch-action: pan-y;
    }

    .home-screen {
      background:
        radial-gradient(circle at 50% 14%, var(--home-center-glow) 0%, transparent 40%),
        linear-gradient(135deg, var(--home-glow-a) 0%, var(--home-glow-b) 100%),
        radial-gradient(circle at 50% 18%, rgba(255, 181, 233, 0.28) 0%, transparent 45%),
        radial-gradient(circle at 82% 74%, rgba(127, 255, 240, 0.12) 0%, transparent 36%),
        var(--home-bg-base);
      transition: background 260ms ease;
    }

    .welcome {
      width: min(700px, 100%);
      min-height: min(680px, 52vh);
      margin: 0 auto;
      border-radius: 26px;
      text-align: center;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 14px;
      padding: 24px;
      background: linear-gradient(180deg, var(--home-card-top), var(--home-card-bottom));
      border: 1px solid rgba(255, 255, 255, 0.26);
      box-shadow: 0 28px 68px rgba(24, 16, 42, 0.2);
      position: relative;
      overflow: hidden;
      transition: background 260ms ease, border-color 260ms ease, box-shadow 260ms ease;
    }

    .welcome::before {
      content: "";
      position: absolute;
      width: 430px;
      height: 430px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(255, 121, 214, 0.36), rgba(255, 121, 214, 0));
      top: -210px;
      left: 50%;
      transform: translateX(-50%);
      pointer-events: none;
    }

    .welcome::after {
      content: "";
      position: absolute;
      inset: auto -15% -60% -15%;
      height: 72%;
      background: radial-gradient(circle at 50% 0%, rgba(255, 210, 239, 0.55) 0%, rgba(255, 210, 239, 0) 72%);
      pointer-events: none;
    }

    .welcome-main {
      position: relative;
      z-index: 1;
      display: grid;
      justify-items: center;
      gap: 14px;
      max-width: 560px;
    }

    .avatar-shell {
      width: 150px;
      height: 150px;
      border-radius: 999px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(255, 255, 255, 0.24);
      background: radial-gradient(circle at 30% 24%, #322f52 0%, #1f1d33 58%, #121321 100%);
      box-shadow: 0 18px 34px rgba(31, 17, 46, 0.34), 0 0 0 2px #6f7dff57, 0 0 0 7px #ff66c429;
      animation: breathe 4.2s ease-in-out infinite;
      position: relative;
      isolation: isolate;
    }

    .avatar-shell::before {
      content: "";
      position: absolute;
      inset: -6px;
      border-radius: inherit;
      background: conic-gradient(from 210deg, #ff66c4, #8e7dff, #7ffff0, #ffcf6c, #ff66c4);
      opacity: 0.28;
      z-index: -1;
      filter: blur(7px);
    }

    .avatar-bot {
      width: 88px;
      height: 88px;
      filter: drop-shadow(0 8px 18px rgba(0, 0, 0, 0.32));
    }

    .welcome h1 {
      margin: 0;
      font-family: "Outfit", "Manrope", sans-serif;
      font-size: clamp(34px, 6.5vw, 46px);
      line-height: 1.02;
      letter-spacing: -0.7px;
      font-weight: 700;
      background: linear-gradient(180deg, color-mix(in srgb, var(--home-title) 96%, #ffffff), color-mix(in srgb, var(--home-title) 76%, #6d62a2));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      text-wrap: balance;
      transition: filter 220ms ease;
      filter: drop-shadow(0 1px 0 rgba(255, 255, 255, 0.2));
    }

    .welcome p {
      margin: 0;
      color: var(--home-sub);
      font-size: clamp(15px, 2.2vw, 18px);
      line-height: 1.45;
      max-width: 520px;
      text-wrap: balance;
      font-family: "Outfit", "Manrope", sans-serif;
      font-weight: 500;
      letter-spacing: -0.01em;
      transition: color 220ms ease;
    }

    .slide-hint {
      position: relative;
      z-index: 1;
      color: var(--home-hint);
      font-weight: 700;
      font-size: 14px;
      letter-spacing: 0.01em;
      opacity: 0;
      animation: hintIn 700ms ease 250ms forwards, hintBounce 1.7s ease-in-out 1.1s infinite;
      cursor: pointer;
      user-select: none;
      margin-top: 2px;
      transition: color 220ms ease;
    }

    .chat-wrap {
      width: min(700px, 100%);
      display: grid;
      gap: 12px;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      opacity: 0;
      animation: rise 420ms ease-out 120ms forwards;
    }

    .brand {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: color-mix(in srgb, var(--panel) 78%, transparent);
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-weight: 700;
    }

    .top-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .nav-btn {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 700;
      color: var(--ink);
      background: var(--panel);
      cursor: pointer;
      min-width: 0;
    }

    .eval-link {
      text-decoration: none;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 700;
      color: var(--ink);
      background: var(--panel);
    }

    .theme-toggle {
      border: 1px solid var(--line);
      background: linear-gradient(145deg, color-mix(in srgb, var(--panel) 92%, #fff), color-mix(in srgb, var(--bg-soft) 80%, var(--panel)));
      color: var(--ink);
      border-radius: 999px;
      width: 42px;
      height: 42px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: var(--shadow);
      transition: transform 160ms ease, border-color 180ms ease, background 220ms ease;
      position: relative;
      overflow: hidden;
    }

    .theme-toggle:hover {
      transform: translateY(-1px) scale(1.03);
      border-color: color-mix(in srgb, var(--brand) 38%, var(--line));
    }

    .theme-toggle svg {
      width: 18px;
      height: 18px;
      position: absolute;
      stroke: currentColor;
      fill: none;
      stroke-width: 1.8;
      transition: opacity 220ms ease, transform 260ms ease;
    }

    .theme-toggle .moon-icon {
      opacity: 0;
      transform: scale(0.7) rotate(-20deg);
    }

    body[data-theme="dark"] .theme-toggle .sun-icon {
      opacity: 0;
      transform: scale(0.7) rotate(20deg);
    }

    body[data-theme="dark"] .theme-toggle .moon-icon {
      opacity: 1;
      transform: scale(1) rotate(0deg);
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-height: min(78vh, 760px);
      position: relative;
      opacity: 0;
      animation: rise 620ms ease-out 180ms forwards;
    }

    .panel::before {
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 3px;
      background: linear-gradient(90deg, color-mix(in srgb, var(--brand) 65%, transparent), color-mix(in srgb, #f0b98f 58%, transparent));
      opacity: 0.65;
      pointer-events: none;
    }

    .head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 15px 18px;
      border-bottom: 1px solid var(--line);
      background: color-mix(in srgb, var(--bg-soft) 90%, transparent);
    }

    .head h2 {
      margin: 0;
      font-size: 16px;
      font-weight: 800;
    }

    .status {
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      color: color-mix(in srgb, var(--brand-strong) 80%, #163e38);
      background: color-mix(in srgb, var(--user) 72%, var(--panel));
      border: 1px solid color-mix(in srgb, var(--brand) 24%, var(--line));
      border-radius: 999px;
      padding: 6px 11px;
      letter-spacing: 0.04em;
    }

    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      white-space: pre-wrap;
      line-height: 1.58;
      scroll-behavior: auto;
      overscroll-behavior: contain;
    }

    #chat::-webkit-scrollbar {
      width: 8px;
    }

    #chat::-webkit-scrollbar-thumb {
      background: color-mix(in srgb, var(--muted) 35%, transparent);
      border-radius: 999px;
    }

    .msg {
      margin-bottom: 12px;
      padding: 12px 14px;
      border-radius: 18px;
      border: 1px solid var(--line);
      animation: pop 260ms ease-out;
      max-width: 88%;
      font-size: 14px;
      box-shadow: 0 7px 16px rgba(16, 21, 32, 0.06);
      transition: transform 120ms ease, box-shadow 120ms ease;
    }

    .msg:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 18px rgba(16, 21, 32, 0.09);
    }

    .user {
      margin-left: auto;
      background: var(--user);
      border-color: color-mix(in srgb, var(--brand) 24%, var(--line));
    }

    .bot {
      background: var(--bot);
    }

    .msg-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 6px;
      font-size: 11px;
      color: var(--muted);
      font-family: "IBM Plex Mono", monospace;
    }

    .copy-btn {
      opacity: 0;
      border: 1px solid var(--line);
      background: color-mix(in srgb, var(--panel) 76%, transparent);
      color: var(--muted);
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 999px;
      min-width: 0;
      transition: opacity 160ms ease, transform 130ms ease;
      transform: translateY(1px);
    }

    .msg:hover .copy-btn {
      opacity: 1;
      transform: translateY(0);
    }

    .msg-content h1, .msg-content h2, .msg-content h3 {
      margin: 0 0 8px;
      line-height: 1.25;
    }

    .msg-content p {
      margin: 0 0 8px;
    }

    .msg-content ul, .msg-content ol {
      margin: 0 0 8px 18px;
      padding: 0;
    }

    .msg-content code {
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      background: color-mix(in srgb, var(--line) 35%, transparent);
      padding: 2px 5px;
      border-radius: 6px;
    }

    .msg-content pre {
      margin: 8px 0;
      padding: 10px;
      border-radius: 10px;
      overflow-x: auto;
      background: color-mix(in srgb, var(--line) 32%, transparent);
    }

    .sources {
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px dashed color-mix(in srgb, var(--muted) 30%, transparent);
      color: var(--muted);
      font-size: 11px;
      font-family: "IBM Plex Mono", monospace;
      line-height: 1.45;
    }

    .sources strong {
      display: inline-block;
      margin-bottom: 3px;
      letter-spacing: 0.04em;
      text-transform: none;
    }

    .sources .sep {
      display: block;
      margin-bottom: 4px;
      color: color-mix(in srgb, var(--muted) 85%, transparent);
    }

    .typing {
      display: inline-flex;
      gap: 5px;
      align-items: center;
      min-height: 16px;
    }

    .typing span {
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: color-mix(in srgb, var(--brand) 72%, var(--muted));
      animation: dotBounce 900ms infinite ease-in-out;
    }

    .typing span:nth-child(2) { animation-delay: 130ms; }
    .typing span:nth-child(3) { animation-delay: 260ms; }

    .composer {
      display: flex;
      gap: 10px;
      padding: 12px;
      border-top: 1px solid var(--line);
      background: color-mix(in srgb, var(--bg-soft) 86%, transparent);
      position: sticky;
      bottom: 0;
      backdrop-filter: none;
    }

    @media (prefers-reduced-motion: reduce) {
      .flow,
      .panel,
      .topbar,
      .msg,
      .avatar-shell,
      .slide-hint,
      .typing span {
        animation: none !important;
        transition: none !important;
      }
    }

    input[type="text"] {
      flex: 1;
      padding: 13px 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      font-size: 14px;
      outline: none;
      font-family: "Manrope", sans-serif;
      background: color-mix(in srgb, var(--panel) 94%, transparent);
      color: var(--ink);
      transition: border-color 180ms ease, box-shadow 180ms ease;
    }

    input[type="text"]:focus {
      border-color: color-mix(in srgb, var(--brand) 52%, var(--line));
      box-shadow: var(--input-shadow);
    }

    button {
      border: none;
      border-radius: 16px;
      padding: 0 16px;
      background: var(--brand);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
      min-width: 96px;
      transition: transform 130ms ease, background 170ms ease, filter 160ms ease;
      font-family: "Manrope", sans-serif;
    }

    button:hover {
      background: var(--brand-strong);
      transform: translateY(-1px) scale(1.02);
      filter: saturate(1.06);
    }

    button:active {
      transform: scale(0.98);
    }

    button:disabled {
      opacity: 0.65;
      cursor: not-allowed;
      transform: none;
    }

    .error {
      color: var(--danger);
      font-size: 12px;
      padding: 0 14px 12px;
      min-height: 16px;
    }

    .hint-note {
      color: var(--muted);
      font-size: 12px;
      padding: 0 14px 12px;
    }

    .quick-prompts {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      padding: 0 12px 12px;
    }

    .quick {
      min-width: 0;
      padding: 10px;
      border: 1px solid color-mix(in srgb, var(--brand) 28%, var(--line));
      color: color-mix(in srgb, var(--brand-strong) 76%, var(--ink));
      background: color-mix(in srgb, var(--user) 70%, #fffdf8);
      border-radius: 12px;
      font-size: 12px;
      line-height: 1.35;
      text-align: left;
      font-weight: 600;
      min-height: 0;
    }

    .quick:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 14px rgba(14, 45, 40, 0.08);
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pop {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes breathe {
      0%, 100% { transform: translateY(0) scale(1); }
      50% { transform: translateY(-6px) scale(1.02); }
    }

    @keyframes hintIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes hintBounce {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-4px); }
    }

    @keyframes dotBounce {
      0%, 80%, 100% { transform: translateY(0); opacity: 0.7; }
      40% { transform: translateY(-4px); opacity: 1; }
    }

    @media (max-width: 520px) {
      .welcome {
        border-radius: 20px;
        padding: 18px 14px;
        min-height: min(560px, 86vh);
      }

      .welcome h1 {
        font-size: clamp(30px, 10vw, 38px);
      }

      .welcome p {
        font-size: clamp(14px, 4vw, 16px);
      }

      .avatar-shell {
        width: 128px;
        height: 128px;
      }

      .avatar-bot {
        width: 76px;
        height: 76px;
      }

      .panel {
        min-height: 76vh;
      }

      .composer {
        flex-direction: column;
      }

      .quick-prompts {
        grid-template-columns: 1fr;
      }

      button {
        width: 100%;
        min-height: 44px;
      }

      .msg {
        max-width: 94%;
      }
    }
  </style>
</head>
<body>
  <div id="flow" class="flow">
    <section id="homeScreen" class="screen home-screen">
      <div class="welcome">
        <div class="welcome-main">
          <div class="avatar-shell" aria-hidden="true">
            <svg class="avatar-bot" viewBox="0 0 96 96" fill="none" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="ringGrad" x1="15" y1="15" x2="80" y2="80" gradientUnits="userSpaceOnUse">
                  <stop stop-color="#7E8BFF"/>
                  <stop offset="0.55" stop-color="#9E7DFF"/>
                  <stop offset="1" stop-color="#FF74D1"/>
                </linearGradient>
              </defs>
              <circle cx="48" cy="48" r="35" fill="#171829"/>
              <circle cx="48" cy="48" r="31" stroke="url(#ringGrad)" stroke-width="3"/>
              <rect x="27" y="35" width="42" height="29" rx="13.5" fill="#272743"/>
              <ellipse cx="40.5" cy="49" rx="6" ry="5.6" fill="#7FFFF0"/>
              <ellipse cx="55.5" cy="49" rx="6" ry="5.6" fill="#FF92E7"/>
              <rect x="38" y="61.5" width="20" height="3.8" rx="1.9" fill="#E8E3FF"/>
              <rect x="46" y="18" width="4" height="12" rx="2" fill="#CFC6F3"/>
              <circle cx="48" cy="15.5" r="4.2" fill="#FFE475"/>
            </svg>
          </div>
          <h1>NUST Registration Copilot</h1>
          <p>Get instant answers for NUST admissions, registration, and student services.</p>
        </div>

        <div id="slideHint" class="slide-hint">Click or Slide Up to Start</div>
      </div>
    </section>

    <section id="chatScreen" class="screen">
      <div class="chat-wrap">
        <div class="topbar">
          <div class="brand">NUST Local RAG</div>
          <div class="top-actions">
            <button id="backHome" class="nav-btn" type="button" style="display:none;">Home</button>
            <a class="eval-link" href="/evaluation">Model Evaluation</a>
            <button id="themeToggle" class="theme-toggle" aria-label="Toggle theme">
              <svg class="sun-icon" viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="12" r="4"></circle>
                <path d="M12 2v3M12 19v3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M2 12h3M19 12h3M4.9 19.1 7 17M17 7l2.1-2.1"></path>
              </svg>
              <svg class="moon-icon" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4 7 7 0 0 0 20 14.5z"></path>
              </svg>
            </button>
          </div>
        </div>

        <div class="panel">
          <div class="head">
            <h2>NUST Administration Support Assistant</h2>
            <span id="status" class="status">READY</span>
          </div>
          <div id="chat"></div>
          <div class="composer">
            <input id="q" type="text" placeholder="Ask e.g. What is the eligibility criteria for UG admissions?" />
            <button id="send">Send</button>
          </div>
          <div id="error" class="error"></div>
          <div class="hint-note">Off-topic questions are refused: I can only help with NUST administration topics.</div>
          <div class="quick-prompts">
            <button class="quick" data-q="What is the registration process for new students at NUST?">Registration process for new students</button>
            <button class="quick" data-q="What documents are required for NUST registration?">Required registration documents</button>
            <button class="quick" data-q="How can I check my fee challan and payment status?">Fee challan and payment status</button>
            <button class="quick" data-q="When does semester course registration open?">Course registration opening dates</button>
          </div>
        </div>
      </div>
    </section>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.7/dist/purify.min.js"></script>

  <script>
    const flow = document.getElementById("flow");
    const homeScreen = document.getElementById("homeScreen");
    const chatScreen = document.getElementById("chatScreen");
    const slideHint = document.getElementById("slideHint");
    const chat = document.getElementById("chat");
    const input = document.getElementById("q");
    const sendBtn = document.getElementById("send");
    const quickButtons = Array.from(document.querySelectorAll(".quick"));
    const status = document.getElementById("status");
    const errorBox = document.getElementById("error");
    const themeToggle = document.getElementById("themeToggle");
    const backHomeBtn = document.getElementById("backHome");
    const SESSION_STORAGE_KEY = "nust_chat_session_id";
    let activeController = null;
    let chatOpened = false;
    let transitionBusy = false;
    let introShown = false;
    let touchStartY = null;

    function setTheme(theme) {
      document.body.setAttribute("data-theme", theme);
      localStorage.setItem("nust-chat-theme", theme);
    }

    setTheme(localStorage.getItem("nust-chat-theme") || "light");

    themeToggle.addEventListener("click", () => {
      const current = document.body.getAttribute("data-theme") || "light";
      setTheme(current === "light" ? "dark" : "light");
    });

    function nowStamp() {
      return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }

    function setStatus(text) {
      status.textContent = text;
    }

    function scrollToLatest() {
      chat.scrollTop = chat.scrollHeight;
    }

    function addMessage(cssClass, text, label = "") {
      const node = document.createElement("div");
      node.className = `msg ${cssClass}`;

      const header = document.createElement("div");
      header.className = "msg-head";

      const who = document.createElement("span");
      who.textContent = `${label} ${nowStamp()}`.trim();

      const copyBtn = document.createElement("button");
      copyBtn.className = "copy-btn";
      copyBtn.type = "button";
      copyBtn.textContent = "Copy";
      copyBtn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(text);
          copyBtn.textContent = "Copied";
          setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200);
        } catch (err) {
          copyBtn.textContent = "Failed";
          setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200);
        }
      });

      header.appendChild(who);
      header.appendChild(copyBtn);
      node.appendChild(header);

      const body = document.createElement("div");
      body.className = "msg-content";
      if (cssClass === "bot" && window.marked) {
        const rawHtml = marked.parse(text || "");
        body.innerHTML = window.DOMPurify ? DOMPurify.sanitize(rawHtml) : rawHtml;
      } else {
        body.textContent = text;
      }
      node.appendChild(body);

      chat.appendChild(node);
      scrollToLatest();
      return node;
    }

    function addTypingIndicator() {
      const node = document.createElement("div");
      node.className = "msg bot";

      const header = document.createElement("div");
      header.className = "msg-head";
      header.textContent = `NUST Assistant ${nowStamp()}`;
      node.appendChild(header);

      const dots = document.createElement("div");
      dots.className = "typing";
      dots.innerHTML = "<span></span><span></span><span></span>";
      node.appendChild(dots);

      chat.appendChild(node);
      scrollToLatest();
      return node;
    }

    function addSources(container, sources) {
      if (!Array.isArray(sources) || !sources.length) return;
      const src = document.createElement("div");
      src.className = "sources";
      src.innerHTML = `<span class="sep">---</span><strong>Sources:</strong><br>${sources.map((s) => `• ${s}`).join("<br>")}`;
      container.appendChild(src);
    }

    function extractAnswerAndSources(fullText) {
      const marker = "\\n\\nSources:\\n- ";
      const idx = fullText.indexOf(marker);
      if (idx < 0) {
        return { answer: fullText.trim() || "No answer returned.", sources: [] };
      }
      const answer = fullText.slice(0, idx).trim() || "No answer returned.";
      const srcText = fullText.slice(idx + marker.length).trim();
      const sources = srcText ? srcText.split("\\n- ").map((s) => s.trim()).filter(Boolean) : [];
      return { answer, sources };
    }

    async function ask(question, signal) {
      errorBox.textContent = "";
      addMessage("user", question, "You");
      setStatus("THINKING");
      const typingNode = addTypingIndicator();

      const streamNode = document.createElement("div");
      streamNode.className = "msg bot";
      streamNode.innerHTML = `<div class="msg-head">NUST Assistant ${nowStamp()}</div><div class="msg-content"></div>`;
      const streamBody = streamNode.querySelector(".msg-content");
      let renderedAnswer = "";

      try {
        const sessionId = localStorage.getItem(SESSION_STORAGE_KEY) || "";
        const streamUrl = `/chat/stream?message=${encodeURIComponent(question)}&session_id=${encodeURIComponent(sessionId)}`;

        const res = await fetch(streamUrl, {
          method: "GET",
          signal
        });

        if (!res.ok || !res.body) {
          throw new Error("Streaming request failed with status " + res.status);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let full = "";
        let streamStarted = false;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split("\\n\\n");
          buffer = events.pop() || "";

          for (const evt of events) {
            if (!evt.startsWith("data: ")) continue;
            const data = evt.slice(6);

            if (data.startsWith("[SESSION]")) {
              const incomingSessionId = data.slice("[SESSION]".length).trim();
              if (incomingSessionId) {
                localStorage.setItem(SESSION_STORAGE_KEY, incomingSessionId);
              }
              continue;
            }

            if (data === "[START]") {
              setStatus("STREAMING");
              if (!streamStarted) {
                streamStarted = true;
                typingNode.remove();
                chat.appendChild(streamNode);
                scrollToLatest();
              }
              continue;
            }
            if (data === "[END]") {
              continue;
            }

            full += data;
            const parsedNow = extractAnswerAndSources(full);
            renderedAnswer = parsedNow.answer;
            if (window.marked) {
              const rawHtml = marked.parse(renderedAnswer || "");
              streamBody.innerHTML = window.DOMPurify ? DOMPurify.sanitize(rawHtml) : rawHtml;
            } else {
              streamBody.textContent = renderedAnswer;
            }
            scrollToLatest();
          }
        }

        if (typingNode.isConnected) {
          typingNode.remove();
        }
        const parsed = extractAnswerAndSources(full);
        if (streamNode.isConnected) {
          if (streamBody) {
            if (window.marked) {
              const rawHtml = marked.parse(parsed.answer || "");
              streamBody.innerHTML = window.DOMPurify ? DOMPurify.sanitize(rawHtml) : rawHtml;
            } else {
              streamBody.textContent = parsed.answer;
            }
          }
          addSources(streamNode, parsed.sources);
        } else {
          const botNode = addMessage("bot", parsed.answer, "NUST Assistant");
          addSources(botNode, parsed.sources);
        }
      } catch (streamErr) {
        if (typingNode.isConnected) {
          typingNode.remove();
        }
        if (streamNode.isConnected && !renderedAnswer.trim()) {
          streamNode.remove();
        }
        throw streamErr;
      }

      setStatus("READY");
    }

    async function onSend() {
      if (sendBtn.disabled) return;
      const question = input.value.trim();
      if (!question) return;

      input.value = "";
      sendBtn.disabled = true;
      input.disabled = true;
      activeController = new AbortController();
      const timeoutId = setTimeout(() => {
        if (activeController) {
          activeController.abort();
        }
      }, 120000);

      try {
        await ask(question, activeController.signal);
      } catch (err) {
        setStatus("ERROR");
        errorBox.textContent = "Request timed out or failed. Please retry with a shorter question.";
        addMessage("bot", "Something went wrong while processing your request.", "NUST Assistant");
      } finally {
        clearTimeout(timeoutId);
        activeController = null;
        sendBtn.disabled = false;
        input.disabled = false;
        input.focus();
      }
    }

    async function showChatIntro() {
      if (introShown) return;
      introShown = true;
      const typingNode = addTypingIndicator();
      await new Promise((resolve) => setTimeout(resolve, 900));
      typingNode.remove();
      addMessage("bot", "Hi! How can I help you today?", "NUST Assistant");
      setStatus("READY");
    }

    function openChatScreen() {
      if (chatOpened || transitionBusy) return;
      transitionBusy = true;
      chatOpened = true;
      flow.classList.add("chat-open");
      if (backHomeBtn) {
        backHomeBtn.style.display = "inline-flex";
      }
      setTimeout(() => {
        transitionBusy = false;
      }, 560);
      setTimeout(() => {
        input.focus();
      }, 450);
      showChatIntro();
    }

    function closeChatScreen() {
      if (!chatOpened || transitionBusy) return;
      transitionBusy = true;
      chatOpened = false;
      flow.classList.remove("chat-open");
      if (backHomeBtn) {
        backHomeBtn.style.display = "none";
      }
      setTimeout(() => {
        transitionBusy = false;
      }, 560);
    }

    function onWheel(event) {
      if (chatOpened || transitionBusy) return;
      if (event.deltaY > 10) {
        openChatScreen();
      }
    }

    homeScreen.addEventListener("wheel", onWheel, { passive: true });

    homeScreen.addEventListener("touchstart", (event) => {
      const touch = event.changedTouches && event.changedTouches[0];
      touchStartY = touch ? touch.clientY : null;
    }, { passive: true });

    homeScreen.addEventListener("touchend", (event) => {
      if (chatOpened || touchStartY === null || transitionBusy) return;
      const touch = event.changedTouches && event.changedTouches[0];
      if (!touch) return;
      const delta = touchStartY - touch.clientY;
      if (delta > 35) {
        openChatScreen();
      }
      touchStartY = null;
    }, { passive: true });

    window.addEventListener("keydown", (event) => {
      if (!chatOpened && (event.key === "ArrowUp" || event.key === "Enter" || event.key === " ")) {
        openChatScreen();
        return;
      }
      if (chatOpened && (event.key === "ArrowDown" || event.key === "Escape")) {
        closeChatScreen();
      }
    });

    if (slideHint) {
      slideHint.addEventListener("click", openChatScreen);
    }

    if (backHomeBtn) {
      backHomeBtn.addEventListener("click", closeChatScreen);
    }

    quickButtons.forEach((btn) => {
      btn.addEventListener("click", async () => {
        input.value = btn.dataset.q || "";
        await onSend();
      });
    });

    sendBtn.addEventListener("click", onSend);
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") onSend();
    });
  </script>
</body>
</html>
    """
