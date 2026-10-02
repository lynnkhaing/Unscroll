"""Unscroll: turn assigned readings into a swipeable learn-feed.

Gemini Flash digests the reading (multimodal PDF -> concept reels, streamed one by one).
Gemma 4 (open-weight, via the Gemini API) is the tutor: quizzes, teach-back grading, and
judging whether a Wikimedia Commons image really fits a concept.
Gemini TTS narrates; Gemini image generation (Vertex AI) illustrates only when no real image fits.
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel

DIGEST_MODEL = os.environ.get("DIGEST_MODEL", "gemini-3.1-flash-lite")
# If the primary model is overloaded (503/429), fall back so a live demo never dead-ends.
DIGEST_FALLBACKS = [DIGEST_MODEL] + [m for m in os.environ.get(
    "DIGEST_FALLBACKS", "gemini-3.5-flash,gemini-3.8-flash,gemini-flash-latest").split(",") if m and m != DIGEST_MODEL]
TUTOR_MODEL = os.environ.get("TUTOR_MODEL", "gemma-4-26b-a4b-it")
TTS_MODEL = os.environ.get("TTS_MODEL", "gemini-3.8-flash-tts")
TTS_VOICE = os.environ.get("TTS_VOICE", "Puck")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gemini-3.1-flash-image")
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0340911872")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SKILL_PATH = Path(__file__).parent / "skills" / "unscroll-tutor" / "SKILL.md"
USER_AGENT = "Unscroll/1.0 (https://github.com/lynnkhaing/Unscroll; SFSU hackathon project)"

app = FastAPI(title="Unscroll")
_client = None
_client_lock = threading.Lock()  # clients are shared across threads; never build two (the loser closes itself)


def client() -> genai.Client:
    global _client
    with _client_lock:
        if _client is None:
            if not os.environ.get("GEMINI_API_KEY"):
                raise HTTPException(500, "GEMINI_API_KEY is not set on the server.")
            _client = genai.Client()
        return _client


def vertex() -> genai.Client:
    """Vertex AI client billed to the project's Google Cloud credits (used for image gen + TTS fallback).
    On Cloud Run this uses the service account; locally, Application Default Credentials or
    GCLOUD_ACCESS_TOKEN (from `gcloud auth print-access-token`)."""
    return vertex_at("global")


_vertex_clients: dict[str, genai.Client] = {}


def vertex_at(location: str) -> genai.Client:
    with _client_lock:
        if location not in _vertex_clients:
            kw = dict(vertexai=True, project=GCP_PROJECT, location=location)
            if os.environ.get("GCLOUD_ACCESS_TOKEN"):
                from google.oauth2.credentials import Credentials
                kw["credentials"] = Credentials(os.environ["GCLOUD_ACCESS_TOKEN"])
            _vertex_clients[location] = genai.Client(**kw)
        return _vertex_clients[location]


class LRU(OrderedDict):
    def __init__(self, size: int):
        super().__init__()
        self.size, self.lock = size, threading.Lock()

    def get_(self, k):
        with self.lock:
            if k in self:
                self.move_to_end(k)
                return self[k]
        return None

    def put(self, k, v):
        with self.lock:
            self[k] = v
            if len(self) > self.size:
                self.popitem(last=False)


# ---------- Digest (Gemini), streamed one concept at a time ----------

class Concept(BaseModel):
    title: str
    emoji: str
    script: str
    key_points: list[str]
    source_quote: str
    page: int


DIGEST_PROMPT = """You are turning an assigned college reading into a short-form learning feed
for busy San Francisco State University students (many commute, work, or speak English as a
second language).

Extract the {n} most important concepts, in the order a learner should meet them.

OUTPUT FORMAT: JSON Lines. One compact JSON object per line, nothing else (no markdown, no array,
no blank lines). Write the lines in this order:
1) {{"type":"deck","deck_title":str,"subject":str}}  (a catchy title for the whole reading + the academic subject)
2) then {n} lines, one per concept, each:
{{"type":"concept","title":str,"emoji":str,"script":str,"beats":[beat],"key_points":[str],"source_quote":str,"page":int,"figure_id":str,"image_query":str,"image_prompt":str}}
EVERY concept line MUST include "beats" (3-5 beats), not just the first one.

Field rules:
- title: 2-6 words. emoji: one emoji that fits.
- script: a 30-second narration (55-80 words). Open with a hook (question or surprising fact),
  explain in plain language at a 9th-grade reading level, end with why it matters.
  No filler, no "In this reading". Only use facts that are in the source.
- key_points: 2-4 short factual points from the source a student must remember.
- source_quote: a short verbatim quote (under 25 words) from the source supporting this concept.
- page: the page number in the source where it appears (1 if unknown or plain text).
- figure_id: the id of ONE figure from the FIGURES list below that directly illustrates this
  concept, or "" if none fits. Never reuse a figure for two concepts. Do not force a match.
- image_query: 2-4 words to search Wikimedia Commons for a REAL photo or standard diagram that
  literally shows this concept (a named person, place, event, organism, object, or well-known
  diagram, e.g. "Kitty Genovese", "mitochondria diagram", "Golden Gate Bridge"). Use "" when the
  concept is abstract and no literal real-world image exists (then we illustrate it instead).
- image_prompt: one sentence describing a vivid, accurate, engaging illustration of the concept
  (a concrete scene or metaphor; no words, letters, or numbers in the image).

- beats: a storyboard of 3-5 animated scenes. Start a NEW beat only when the narration moves to a
  different idea; never split one idea across beats. Every beat's visual must show exactly what is
  being said during it that play while the script is narrated, like a
  short explainer video. Each beat: {{"say": the exact consecutive words of the script spoken
  during this scene (beats in order, together covering the whole script), "scene": one of the
  scene types below, plus that scene's fields}}. Make it feel like a motion-graphics explainer:
  use AT MOST 2 "image" beats per reel, and AT LEAST one "diagram", "compare", "quote", "stat", or "term"
  beat. Start with "image" or "term".
  Example beat: {{"say":"Alone, 85% helped; with four others, only 31% did.","scene":"stat","value":"85%","label":"helped when alone","value2":"31%","label2":"with four others"}}
  ("scene" is always a plain string; the scene's fields sit next to it.)
  Scene types (ALL text and numbers must come from the source; never invent facts):
  * "image": {{"focus": "center"|"top"|"bottom"|"left"|"right", "image_prompt": str}}  a picture of
    exactly what this beat says (image_prompt describes it: a concrete scene, no text in the image)
  * "stat": {{"value": str, "label": str, "value2": str, "label2": str}}  a number from the source
    (e.g. "85%"), optional second number to contrast ("" if none). Only if the source has numbers.
  * "diagram": {{"steps": [str]}}  2-5 short labels (1-3 words) of a process/sequence/cause chain
  * "compare": {{"left_title": str, "left": str, "right_title": str, "right": str}}  two short sides
  * "quote": {{"text": str}}  a verbatim excerpt (under 20 words) copied exactly from the source
  * "term": {{"term": str, "definition": str}}  a key term and a definition under 12 words

FIGURES (cropped from the source document):
{figures}
"""


# ---------- Source figures (real visuals from the reading, never generated) ----------

CAPTION_RE = re.compile(r"\s*(Figure|Fig\.|Table|Chart|Exhibit)\s*\d+([.\-]\d+)?", re.I)


def extract_figures(pdf_bytes: bytes, limit: int = 16) -> list[dict]:
    """Find captioned figures in a PDF and crop them (drawings + images + caption) to PNG."""
    figs: list[dict] = []
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return figs
    for page in doc:
        if len(figs) >= limit:
            break
        blocks = [b for b in page.get_text("blocks") if CAPTION_RE.match(b[4])]
        if not blocks:
            continue
        try:
            shapes = [r for r in page.cluster_drawings() if r.width * r.height > 3000]
        except Exception:
            shapes = []
        for img in page.get_images():
            try:
                shapes += [r for r in page.get_image_rects(img[0]) if r.width * r.height > 3000]
            except Exception:
                pass
        for b in blocks:
            cap = pymupdf.Rect(b[:4])
            # Graphics that sit just above (or around) the caption belong to this figure.
            near = [r for r in shapes if r.y1 <= cap.y1 + 4 and cap.y0 - r.y1 < 60
                    and r.x1 > cap.x0 and r.x0 < cap.x1]
            if not near:
                continue
            region = pymupdf.Rect(cap)
            for r in near:
                region |= r
            for r in shapes:  # pull in pieces of the same figure
                if r.intersects(region):
                    region |= r
            if region.height < 40:
                continue
            region = (region + (-6, -6, 6, 6)) & page.rect
            png = page.get_pixmap(clip=region, dpi=144).tobytes("png")
            caption = " ".join(b[4].split())[:220]
            figs.append({
                "id": f"F{len(figs) + 1}",
                "page": page.number + 1,
                "caption": caption,
                "data_url": "data:image/png;base64," + base64.b64encode(png).decode(),
            })
    return figs


def extract_text(pdf_bytes: bytes, max_chars: int = 120_000) -> str:
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return ""
    return "\n".join(f"[page {i + 1}]\n{p.get_text()}" for i, p in enumerate(doc))[:max_chars]


# ---------- Web links ----------

class _TextGrab(__import__("html.parser").parser.HTMLParser):
    KEEP = {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "figcaption", "td", "th", "pre"}
    SKIP = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"}

    def __init__(self):
        super().__init__()
        self.out, self.stack, self.skip, self.title, self._t = [], [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag == "title":
            self._t = True
        if tag in self.KEEP:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self._t = False

    def handle_data(self, data):
        if self._t:
            self.title += data
        elif not self.skip and data.strip():
            self.out.append(data)


def _public_url(url: str) -> bool:
    import ipaddress, socket
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    try:
        for info in socket.getaddrinfo(u.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
    except Exception:
        return False
    return True


def fetch_url(url: str) -> tuple[bytes, str]:
    if not _public_url(url):
        raise HTTPException(400, "Please use a public http(s) link.")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT + " Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            if not _public_url(r.geturl()):
                raise HTTPException(400, "That link redirects somewhere we can't open.")
            return r.read(MAX_UPLOAD_BYTES), (r.headers.get_content_type() or "")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Couldn't open that link ({str(e)[:80]}).")


def html_to_text(html: bytes) -> tuple[str, str]:
    g = _TextGrab()
    g.feed(html.decode("utf-8", errors="ignore"))
    text = re.sub(r"\n\s*\n+", "\n\n", "".join(g.out)).strip()
    return " ".join(g.title.split()), text[:120_000]


@app.post("/api/ingest")
async def ingest(file: UploadFile | None = File(None), text: str = Form(""), url: str = Form(""), n: int = Form(8)):
    n = max(3, min(n, 12))
    parts: list = []
    figures: list[dict] = []
    if url.strip() and not (file is not None and file.filename):
        data, ctype = await asyncio.to_thread(fetch_url, url.strip())
        if ctype == "application/pdf" or data[:4] == b"%PDF":
            figures = await asyncio.to_thread(extract_figures, data)
            text = await asyncio.to_thread(extract_text, data)
        else:
            title, body = await asyncio.to_thread(html_to_text, data)
            text = f"{title}\n(Source: {url.strip()})\n\n{body}"
        if len(text.strip()) < 200:
            raise HTTPException(400, "Couldn't find enough readable text at that link.")
    if file is not None and file.filename:
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (20 MB max).")
        mime = file.content_type or "application/pdf"
        pdf_text = ""
        if mime == "application/pdf":
            figures = await asyncio.to_thread(extract_figures, data)
            pdf_text = await asyncio.to_thread(extract_text, data)
        if mime == "text/plain":
            parts.append(data.decode("utf-8", errors="ignore"))
        elif len(pdf_text) > 500:
            # Text with page markers is ~10x faster to first token than sending the PDF itself;
            # figures are already cropped separately, so no visual information is lost.
            parts.append(f"SOURCE DOCUMENT (text, with page markers):\n{pdf_text}")
        else:  # scanned PDF or photo of a page: let Gemini read it multimodally
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    elif len(text.strip()) >= 200:
        parts.append(f"SOURCE TEXT:\n{text.strip()}")
    else:
        raise HTTPException(400, "Upload a file or paste at least a paragraph of text.")

    fig_list = "\n".join(f"- {f['id']} (p.{f['page']}): {f['caption']}" for f in figures) or "(none)"
    parts.append(DIGEST_PROMPT.format(n=n, figures=fig_list))
    source = next((p for p in parts if isinstance(p, str)), "")
    return StreamingResponse(stream_digest(parts, figures, n, source), media_type="application/x-ndjson")


async def stream_digest(parts: list, figures: list[dict], n: int, source: str = ""):
    """NDJSON stream: deck line, then each concept as soon as Gemini finishes writing it.
    Pings every few seconds keep slow connections (and Safari) from timing out."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    put = lambda item: loop.call_soon_threadsafe(queue.put_nowait, item)
    threading.Thread(target=_digest_worker, args=(parts, figures, n, put, source), daemon=True).start()
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=4)
        except asyncio.TimeoutError:
            yield '{"type":"ping"}\n'
            continue
        if item is None:
            break
        yield json.dumps(item) + "\n"


def _digest_worker(parts, figures, n, put, source=""):
    src_norm = _norm(source)
    by_id = {f["id"]: f for f in figures}
    used: set[str] = set()
    sent = 0
    errors = []
    # Vertex AI (paid via Cloud credits) answers in ~1s; the free-tier API key can be deprioritized
    # under load, so it's the fallback.
    routes = [(vertex, m) for m in DIGEST_FALLBACKS[:2]] + [(client, m) for m in DIGEST_FALLBACKS]
    for make, model in routes:
        for thinking in (types.ThinkingConfig(thinking_level="minimal"), types.ThinkingConfig(thinking_level="low"), None):
            try:
                buf, deck_sent = "", False
                cfg = types.GenerateContentConfig(temperature=0.4, thinking_config=thinking)
                for chunk in make().models.generate_content_stream(model=model, contents=parts, config=cfg):
                    if os.environ.get("DEBUG_RAW"):
                        print("RAW", model, repr((chunk.text or "")[:120]), flush=True)
                    buf += chunk.text or ""
                    *lines, buf = buf.split("\n")
                    for line in lines:
                        item = _parse_line(line)
                        if not item:
                            continue
                        if item.get("type") == "deck" and not deck_sent:
                            deck_sent = True
                            if sent == 0:
                                put({"type": "deck", "deck_title": item.get("deck_title", "Your reading"),
                                     "subject": item.get("subject", ""), "model": model,
                                     "figures_found": len(figures)})
                        elif item.get("type") == "concept" and sent < n:
                            c = _clean_concept(item, by_id, used, src_norm)
                            if c:
                                if sent == 0 and not deck_sent:
                                    put({"type": "deck", "deck_title": "Your reading", "subject": "",
                                         "model": f"{make.__name__}:{model}", "figures_found": len(figures)})
                                    deck_sent = True
                                put({"type": "concept", "index": sent, "concept": c})
                                sent += 1
                tail = _parse_line(buf)
                if tail and tail.get("type") == "concept" and sent < n:
                    c = _clean_concept(tail, by_id, used, src_norm)
                    if c:
                        put({"type": "concept", "index": sent, "concept": c})
                        sent += 1
                if sent:
                    put({"type": "done", "count": sent})
                    put(None)
                    return
                errors.append(f"{model}: no concepts")
                break
            except Exception as e:
                msg = str(e)
                if sent:  # partial deck is still useful; finish with what we have
                    put({"type": "done", "count": sent, "warning": msg[:200]})
                    put(None)
                    return
                if thinking is not None and "hinking" in msg:
                    continue  # model doesn't support thinking_level; retry without it
                errors.append(f"{model}: {msg[:120]}")
                print("digest error", make.__name__, model, msg[:200], flush=True)
                break
    put({"type": "error", "detail": "Gemini is busy right now, please retry. " + " | ".join(errors)})
    put(None)


def _parse_line(line: str):
    line = line.strip().strip(",").strip()
    if not line.startswith("{"):
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


# ---------- Storyboard fact checks: every on-screen word/number must be traceable to the source ----------

SCENES = {"image", "stat", "diagram", "compare", "quote", "term"}
STOP = set("the a an and or of to in on for with by from is are was were be as at that this it its into "
           "than then more less most their they them we you your our not no".split())


def _norm(t: str) -> str:
    t = (t or "").lower().replace("\u2019", "'").replace("-\n", "")
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)  # keep decimal points, drop sentence periods
    t = re.sub(r"[^a-z0-9%.' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _numbers_ok(texts: list[str], src: str) -> bool:
    for t in texts:
        for num in re.findall(r"\d[\d,.]*%?", t or ""):
            core = num.rstrip(".,").replace(",", "")
            if core and core not in src.replace(",", ""):
                return False
    return True


def _stat_in_context(value: str, label: str, src: str, window: int = 160) -> bool:
    """A stat must appear in the source next to what it describes, not just anywhere
    (a bare "3" occurs in every document)."""
    nums = [n.rstrip(".,").replace(",", "") for n in re.findall(r"\d[\d,.]*%?", value or "")]
    stems = [w[:5] for w in _norm(label).split() if len(w) > 3 and w not in STOP]
    plain = src.replace(",", "")
    for n in nums:
        for m in re.finditer(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", plain):
            ctx = plain[max(0, m.start() - window): m.end() + window]
            if not stems or any(st in ctx for st in stems):
                break
        else:
            return False
    return bool(nums)


def _words_ok(texts: list[str], src: str, need: float = 0.6) -> bool:
    words = [w for t in texts for w in _norm(t).split() if len(w) > 3 and w not in STOP]
    if not words:
        return True
    hits = sum(1 for w in words if w in src or w[:5] in src)  # stem match: "helped" ~ "help"
    return hits / len(words) >= need


def _check_beat(b: dict, src: str) -> dict:
    sc = b.get("scene")
    if isinstance(sc, dict):  # model sometimes nests: {"scene": {"stat": {...}}} or {"scene": {"type": "stat", ...}}
        key = next((k for k in sc if k in SCENES), None)
        if key and isinstance(sc[key], dict):
            b = {**b, **sc[key], "scene": key}
        else:
            b = {**b, **sc, "scene": sc.get("type") or sc.get("scene")}
    scene = b.get("scene") if isinstance(b.get("scene"), str) and b.get("scene") in SCENES else "image"
    out = {"say": str(b.get("say") or ""), "scene": scene}
    if scene == "image":
        out["focus"] = b.get("focus") if b.get("focus") in ("center", "top", "bottom", "left", "right") else "center"
        out["image_prompt"] = str(b.get("image_prompt") or "")[:300]
        return out
    fields = {
        "stat": ["value", "label", "value2", "label2"], "diagram": ["steps"],
        "compare": ["left_title", "left", "right_title", "right"], "quote": ["text"], "term": ["term", "definition"],
    }[scene]
    for f in fields:
        v = b.get(f)
        if f == "steps":
            out[f] = [str(x)[:40] for x in v][:5] if isinstance(v, list) else []
        else:
            out[f] = str(v or "")[:160]
    texts = out["steps"] if scene == "diagram" else [out[f] for f in fields]
    ok = bool(src)  # without source text we can't verify, so only image scenes survive
    if ok and scene == "stat":
        ok = (_stat_in_context(out["value"], out["label"], src)
              and (not out["value2"] or _stat_in_context(out["value2"], out["label2"] or out["label"], src))
              and _numbers_ok(texts, src) and _words_ok([out["label"], out["label2"]], src, .34))
    elif ok and scene == "quote":
        q = _norm(out["text"])
        ok = len(q) > 12 and q in src
    elif ok and scene == "diagram":
        ok = 2 <= len(out["steps"]) <= 5 and _numbers_ok(texts, src) and _words_ok(texts, src)
    elif ok:
        ok = all(texts[:1]) and _numbers_ok(texts, src) and _words_ok(texts, src, .5)
    return out if ok else {"say": out["say"], "scene": "image", "focus": "center", "dropped": scene}


def _auto_beats(item: dict, script: str) -> list[dict]:
    """Fallback storyboard when the model skips beats: built only from verifiable material."""
    sents = re.split(r"(?<=[.!?])\s+", script.strip())
    third = max(1, len(sents) // 3)
    groups = [" ".join(sents[:third]), " ".join(sents[third:2 * third]), " ".join(sents[2 * third:])]
    kp = [str(k) for k in item.get("key_points") or []]
    middle = {"scene": "quote", "text": str(item.get("source_quote") or "")}
    if len(kp) >= 2:
        middle = {"scene": "diagram", "steps": [" ".join(k.split()[:3]) for k in kp[:4]]}
    return [{"say": groups[0], "scene": "term", "term": str(item.get("title") or ""), "definition": " ".join((kp[:1] or [""])[0].split()[:12])},
            {"say": groups[1], **middle},
            {"say": groups[2], "scene": "quote", "text": str(item.get("source_quote") or "")},
            {"say": "", "scene": "image", "focus": "center"}][:4]


def _storyboard(item: dict, script: str, src: str) -> list[dict]:
    raw = item.get("beats") if isinstance(item.get("beats"), list) else []
    if not raw:
        raw = _auto_beats(item, script)
    beats = [_check_beat(b, src) for b in raw[:6] if isinstance(b, dict)]
    # collapse runs of plain image beats (e.g. from dropped facts) so the reel keeps moving
    merged = []
    for b in beats:
        if merged and b["scene"] == "image" and merged[-1]["scene"] == "image":
            merged[-1]["say"] = (merged[-1]["say"] + " " + b["say"]).strip()
            continue
        merged.append(b)
    beats = merged
    if not beats:
        beats = [{"say": script, "scene": "image", "focus": "center"}]
    # Timing: each beat starts where its words begin in the script (fallback: evenly spaced).
    low, pos = script.lower(), 0
    for k, b in enumerate(beats):
        say = b["say"].strip().lower()[:40]
        at = low.find(say, pos) if say else -1
        b["at"] = at if at >= 0 else round(len(script) * k / len(beats))
        pos = max(pos, b["at"])
    beats[0]["at"] = 0
    for k in range(1, len(beats)):
        beats[k]["at"] = max(beats[k]["at"], beats[k - 1]["at"] + 1)
    return beats


def _clean_concept(item: dict, by_id: dict, used: set, src: str = "") -> dict | None:
    try:
        c = Concept.model_validate({
            **item,
            "key_points": [str(k) for k in item.get("key_points") or []],
            "page": int(item.get("page") or 1),
            "emoji": item.get("emoji") or "💡",
            "source_quote": item.get("source_quote") or "",
        }).model_dump()
    except Exception:
        return None
    c["image_query"] = str(item.get("image_query") or "")[:80]
    c["image_prompt"] = str(item.get("image_prompt") or c["title"])[:400]
    try:
        c["beats"] = _storyboard(item, c["script"], src)
    except Exception as e:  # a malformed storyboard must never cost us the reel
        import traceback
        print("storyboard error", repr(e), traceback.format_exc()[-600:], flush=True)
        c["beats"] = [{"say": c["script"], "scene": "image", "focus": "center", "at": 0}]
    f = by_id.get(str(item.get("figure_id") or "").strip())
    if f and f["id"] not in used:
        used.add(f["id"])
        c["figure"] = {"kind": "source", "page": f["page"], "caption": f["caption"], "src": f["data_url"]}
    return c


# ---------- Visuals: Wikimedia Commons (judged by Gemma 4) -> Gemini image (Vertex AI) ----------

class VisualReq(BaseModel):
    title: str
    script: str
    subject: str = ""
    image_query: str = ""
    image_prompt: str = ""


_visual_cache = LRU(300)
_pool = ThreadPoolExecutor(8)


def _http_get(url: str, timeout: float = 8) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def wikimedia_candidates(query: str, limit: int = 4) -> list[dict]:
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": "6",
        "gsrsearch": f"{query} filetype:bitmap|drawing", "gsrlimit": "10", "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata", "iiurlwidth": "640",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    data = json.loads(_http_get(url))
    pages = sorted((data.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
    out = []
    for p in pages:
        info = (p.get("imageinfo") or [{}])[0]
        if info.get("mime") not in ("image/jpeg", "image/png", "image/svg+xml", "image/webp"):
            continue
        if (info.get("width") or 0) < 300:
            continue
        meta = info.get("extmetadata") or {}
        out.append({
            "title": p["title"].removeprefix("File:").rsplit(".", 1)[0],
            "thumb": info.get("thumburl") or info.get("url"),
            "page_url": info.get("descriptionurl"),
            "author": _strip_html((meta.get("Artist") or {}).get("value", ""))[:80] or "Unknown author",
            "license": (meta.get("LicenseShortName") or {}).get("value", ""),
            "description": _strip_html((meta.get("ImageDescription") or {}).get("value", ""))[:200],
        })
        if len(out) >= limit:
            break
    return out


def pick_wikimedia(req: VisualReq) -> dict | None:
    if not req.image_query.strip():
        return None  # abstract concept: skip straight to an illustration
    cands = wikimedia_candidates(req.image_query)
    if not cands:
        return None
    imgs = list(_pool.map(lambda c: _safe_get(c["thumb"]), cands))
    contents: list = [
        f"TASK: image_fit\nConcept: {req.title} ({req.subject})\nNarration: {req.script}\n\n"
        "Below are candidate images from Wikimedia Commons. Pick the ONE that clearly and accurately "
        "and LITERALLY shows this specific concept for a student (a real photo of the exact person/place/"
        "event/thing, or a correct diagram of it). A loose visual metaphor is NOT a fit. Reject generic, "
        "decorative, off-topic, low-quality, text-heavy, or potentially offensive images. "
        "When in doubt, answer -1 (we will illustrate it instead).\n"
        'Return ONLY JSON: {"best": int index or -1, "reason": str}'
    ]
    idx_map = []
    for i, (c, b) in enumerate(zip(cands, imgs)):
        if not b:
            continue
        mime = "image/png" if b[:4] == b"\x89PNG" else "image/jpeg"
        contents += [f"\n[{len(idx_map)}] {c['title']}: {c['description']}",
                     types.Part.from_bytes(data=b, mime_type=mime)]
        idx_map.append(c)
    if not idx_map:
        return None
    verdict = ask_gemma(contents)
    try:
        best = int(verdict.get("best", -1))
    except (TypeError, ValueError):
        best = -1
    if not 0 <= best < len(idx_map):
        return None
    c = idx_map[best]
    return {"kind": "wikimedia", "src": c["thumb"], "title": c["title"], "author": c["author"],
            "license": c["license"], "page_url": c["page_url"], "reason": verdict.get("reason", "")}


def _safe_get(url: str) -> bytes | None:
    try:
        return _http_get(url)
    except Exception:
        return None


IMAGE_STYLE = ("Bold, colorful, modern flat vector illustration with soft gradients and depth, "
               "ONE single cohesive scene filling the frame (no panels, no collage, no triptych), engaging and friendly, accurate to the concept, diverse people "
               "if people appear, absolutely no text, letters, numbers, or labels. Scene: ")


# New projects get a small per-minute image quota per region+model, so we rotate across several.
IMAGE_ENDPOINTS = [("global", IMAGE_MODEL), ("us-central1", "gemini-2.5-flash-image"),
                   ("us-west1", "gemini-2.5-flash-image"), ("us-east1", "gemini-2.5-flash-image"),
                   ("global", "gemini-2.5-flash-image"), ("us-east4", "gemini-2.5-flash-image")]
_image_slots = threading.Semaphore(4)
_image_rr = [0]


def generate_image(req: VisualReq) -> dict | None:
    prompt = IMAGE_STYLE + (req.image_prompt or f"an illustration of {req.title}")
    cfg = types.GenerateContentConfig(response_modalities=["IMAGE"],
                                      image_config=types.ImageConfig(aspect_ratio="4:3"))
    last = None
    with _image_slots:
        _image_rr[0] += 1
        start = _image_rr[0]
        for attempt in range(len(IMAGE_ENDPOINTS) + 2):
            loc, model = IMAGE_ENDPOINTS[(start + attempt) % len(IMAGE_ENDPOINTS)]
            try:
                resp = vertex_at(loc).models.generate_content(model=model, contents=prompt, config=cfg)
                break
            except Exception as e:  # 429 quota / 404 model not in region: try the next endpoint
                last = e
                if not any(x in str(e) for x in ("429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND")):
                    raise
                if attempt >= len(IMAGE_ENDPOINTS) - 1:
                    time.sleep(4)
        else:
            raise last
    for p in resp.candidates[0].content.parts:
        if p.inline_data and p.inline_data.data:
            pix = pymupdf.Pixmap(p.inline_data.data)
            if pix.width > 900:
                pix.shrink(1)
            if pix.alpha:
                pix = pymupdf.Pixmap(pix, 0)
            jpg = pix.tobytes("jpeg", jpg_quality=82)
            return {"kind": "ai", "src": "data:image/jpeg;base64," + base64.b64encode(jpg).decode(),
                    "model": model}
    return None


@app.post("/api/visual")
def visual(req: VisualReq):
    key = (req.title + "|" + req.image_query).lower()
    hit = _visual_cache.get_(key)
    if hit:
        return hit
    errors = []
    for step in (pick_wikimedia, generate_image):
        try:
            out = step(req)
            if out:
                _visual_cache.put(key, out)
                return out
        except Exception as e:
            errors.append(f"{step.__name__}: {str(e)[:120]}")
    return {"kind": "none", "errors": errors}


# ---------- Narration (Gemini TTS, streamed) ----------

class TTSReq(BaseModel):
    text: str


_tts_cache = LRU(200)  # text hash -> full PCM, so replays and prefetched reels are instant
TTS_RATE = 24000
# Only the narration itself is sent: style instructions in the prompt sometimes get read aloud.
# The upbeat delivery comes from the voice choice (TTS_VOICE) instead.
TTS_STYLE = ""


@app.post("/api/tts")
async def tts(req: TTSReq):
    """Streams raw 16-bit mono PCM (24 kHz) as Gemini TTS produces it: first audio in ~1s instead
    of waiting ~10s for the whole clip. Vertex AI (Cloud credits) first, then the Gemini API key."""
    text = req.text.strip()[:1200]
    if not text:
        raise HTTPException(400, "No text.")
    key = hashlib.sha1(text.encode()).hexdigest()
    headers = {"X-Sample-Rate": str(TTS_RATE), "Cache-Control": "no-store"}
    hit = _tts_cache.get_(key)
    if hit:
        return Response(content=hit, media_type="application/octet-stream", headers=headers)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    put = lambda item: loop.call_soon_threadsafe(queue.put_nowait, item)
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE))),
    )

    def worker():
        pcm = bytearray()
        for make in (vertex, client):
            try:
                for ch in make().models.generate_content_stream(model=TTS_MODEL, contents=TTS_STYLE + text, config=cfg):
                    for part in (ch.candidates[0].content.parts if ch.candidates and ch.candidates[0].content else []):
                        if part.inline_data and part.inline_data.data:
                            pcm += part.inline_data.data
                            put(part.inline_data.data)
                if pcm:
                    _tts_cache.put(key, bytes(pcm))
                    break
            except Exception as e:
                print("tts error", make.__name__, str(e)[:160])
                if pcm:  # already streaming audio; can't restart mid-clip
                    break
        put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def body():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(body(), media_type="application/octet-stream", headers=headers)


# ---------- Tutor (Gemma 4) ----------

def tutor_instructions() -> str:
    """The tutor's rules live in the Agent Skill so the same prompt is reusable elsewhere."""
    try:
        body = SKILL_PATH.read_text()
        return body.split("---", 2)[2].strip() if body.startswith("---") else body
    except OSError:
        return "You are a strict but kind study tutor. Only use the provided source text."


def ask_gemma(prompt, thinking: str = "minimal") -> dict:
    try:
        resp = client().models.generate_content(
            model=TUTOR_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=tutor_instructions(), temperature=0.3,
                thinking_config=types.ThinkingConfig(thinking_level=thinking),
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Gemma error: {e}")
    return extract_json(resp.text or "")


def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    if start == -1:
        raise HTTPException(502, "Tutor did not return JSON.")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        raise HTTPException(502, "Tutor returned malformed JSON.")
    return obj if isinstance(obj, dict) else {"items": obj}


class QuizReq(BaseModel):
    concepts: list[Concept]


@app.post("/api/quiz")
def quiz(req: QuizReq):
    src = "\n\n".join(
        f"[{i}] {c.title} (p.{c.page})\n{c.script}\nKey points: {'; '.join(c.key_points)}"
        for i, c in enumerate(req.concepts)
    )
    prompt = f"""TASK: quiz
For EACH concept below write one SHORT multiple-choice recall question that checks the core idea
(not trivia). The question is at most 12 words. Each option is at most 6 words. 4 options, exactly one correct. Wrong options must be plausible misconceptions
a confused student might actually believe: no joke or absurd options. Vary the position of the correct answer.

Return ONLY JSON: {{"questions": [{{"concept_index": int, "question": str,
"options": [str, str, str, str], "answer_index": int, "explanation": str}}]}}

CONCEPTS:
{src}"""
    out = ask_gemma(prompt)
    qs = out.get("questions") or out.get("items") or []
    clean = []
    for q in qs:
        try:
            opts = [str(o) for o in q["options"]][:4]
            ai = int(q["answer_index"])
            if len(opts) == 4 and 0 <= ai < 4:
                clean.append({
                    "concept_index": int(q.get("concept_index", len(clean))),
                    "question": str(q["question"]),
                    "options": opts,
                    "answer_index": ai,
                    "explanation": str(q.get("explanation", "")),
                })
        except (KeyError, ValueError, TypeError):
            continue
    return {"questions": clean, "model": TUTOR_MODEL}


class TeachReq(BaseModel):
    concept: Concept
    explanation: str


@app.post("/api/teachback")
def teachback(req: TeachReq):
    if len(req.explanation.strip()) < 5:
        raise HTTPException(400, "Write or say a little more first.")
    c = req.concept
    prompt = f"""TASK: teachback
A student is explaining a concept in their own words (Feynman technique).
Grade ONLY against the SOURCE below. Do not reward facts that are not in the source.
Ignore grammar and spelling; English may be the student's second language.

SOURCE (page {c.page}):
Title: {c.title}
{c.script}
Key points: {'; '.join(c.key_points)}
Quote: "{c.source_quote}"

STUDENT EXPLANATION:
\"\"\"{req.explanation.strip()[:2000]}\"\"\"

Coach like a Socratic tutor: do NOT reveal the missing ideas directly in "nudges"; instead ask
1-2 short guiding questions that make the student think their way to them.
Return ONLY JSON: {{"score": int 0-100, "verdict": "nailed it" | "almost" | "not yet",
"got_right": [str], "nudges": [str] (guiding questions, max 15 words each, no answers in them),
"missed": [str] (the missing ideas, shown only if the student asks to reveal), "misconceptions": [str],
"tip": str (one encouraging sentence)}}"""
    out = ask_gemma(prompt, thinking=os.environ.get("GRADER_THINKING", "minimal"))
    try:
        score = max(0, min(100, int(out.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    return {
        "score": score,
        "verdict": out.get("verdict") or ("nailed it" if score >= 80 else "almost" if score >= 50 else "not yet"),
        "got_right": list(out.get("got_right") or []),
        "missed": list(out.get("missed") or []),
        "nudges": list(out.get("nudges") or [])[:2],
        "misconceptions": list(out.get("misconceptions") or []),
        "tip": out.get("tip", ""),
        "page": c.page,
        "source_quote": c.source_quote,
        "model": TUTOR_MODEL,
    }


class NextReq(BaseModel):
    deck_title: str
    subject: str = ""
    concepts: list[str]
    weak: list[str] = []


@app.post("/api/next")
def next_topics(req: NextReq):
    prompt = f"""TASK: next_topics
A student just studied "{req.deck_title}" ({req.subject}). Concepts: {"; ".join(req.concepts[:12])}.
They struggled with: {"; ".join(req.weak[:6]) or "nothing in particular"}.
Suggest 3 topics to learn next: one to shore up a weak spot (if any), and two natural next steps.
Return ONLY JSON: {{"topics": [{{"topic": str (2-6 words), "why": str (max 14 words)}}]}}"""
    out = ask_gemma(prompt)
    topics = [t for t in (out.get("topics") or out.get("items") or []) if isinstance(t, dict) and t.get("topic")][:3]
    return {"topics": [{"topic": str(t["topic"])[:60], "why": str(t.get("why", ""))[:120]} for t in topics], "model": TUTOR_MODEL}


@app.get("/api/health")
def health():
    return {"ok": True, "digest_model": DIGEST_MODEL, "tutor_model": TUTOR_MODEL,
            "key": bool(os.environ.get("GEMINI_API_KEY"))}


STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    # Always revalidate the page: a cached old frontend can't read the newer streaming API.
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache, must-revalidate"})
