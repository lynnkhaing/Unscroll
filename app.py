"""Unscroll: turn assigned readings into a swipeable learn-feed.

Gemini Flash digests the reading (multimodal PDF -> concept reels).
Gemma 4 (open-weight, via the Gemini API) is the tutor: quizzes + teach-back grading.
"""

import base64
import json
import os
import re
from pathlib import Path

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel

DIGEST_MODEL = os.environ.get("DIGEST_MODEL", "gemini-3.8-flash")
# If the primary model is overloaded (503/429), fall back so a live demo never dead-ends.
DIGEST_FALLBACKS = [DIGEST_MODEL] + [m for m in os.environ.get(
    "DIGEST_FALLBACKS", "gemini-3.5-flash,gemini-flash-latest,gemini-3.1-flash-lite").split(",") if m and m != DIGEST_MODEL]
TUTOR_MODEL = os.environ.get("TUTOR_MODEL", "gemma-4-26b-a4b-it")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SKILL_PATH = Path(__file__).parent / "skills" / "unscroll-tutor" / "SKILL.md"

app = FastAPI(title="Unscroll")
_client = None


def client() -> genai.Client:
    global _client
    if _client is None:
        if not os.environ.get("GEMINI_API_KEY"):
            raise HTTPException(500, "GEMINI_API_KEY is not set on the server.")
        _client = genai.Client()
    return _client


# ---------- Digest (Gemini) ----------

class Concept(BaseModel):
    title: str
    emoji: str
    script: str
    key_points: list[str]
    source_quote: str
    page: int


class DeckConcept(Concept):
    figure_id: str


class Deck(BaseModel):
    deck_title: str
    subject: str
    concepts: list[DeckConcept]


DIGEST_PROMPT = """You are turning an assigned college reading into a short-form learning feed
for busy San Francisco State University students (many commute, work, or speak English as a
second language).

Extract the {n} most important concepts, in the order a learner should meet them.
For each concept:
- title: 2-6 words.
- emoji: one emoji that fits.
- script: a 30-second narration (55-80 words). Open with a hook (question or surprising fact),
  explain in plain language at a 9th-grade reading level, end with why it matters.
  No filler, no "In this reading". Only use facts that are in the source.
- key_points: 2-4 short factual points from the source a student must remember.
- source_quote: a short verbatim quote (under 25 words) from the source supporting this concept.
- page: the page number in the source where it appears (1 if unknown or plain text).
- figure_id: the id of ONE figure from the FIGURES list below that directly illustrates this
  concept, or "" if none fits. Never reuse a figure for two concepts. Do not force a match.

deck_title: a catchy title for the whole reading. subject: the academic subject.

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



@app.post("/api/ingest")
async def ingest(file: UploadFile | None = File(None), text: str = Form(""), n: int = Form(8)):
    n = max(3, min(n, 12))
    parts: list = []
    figures: list[dict] = []
    if file is not None and file.filename:
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (20 MB max).")
        mime = file.content_type or "application/pdf"
        if mime == "application/pdf":
            figures = extract_figures(data)
        if mime == "text/plain":
            parts.append(data.decode("utf-8", errors="ignore"))
        else:
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    elif text.strip():
        parts.append(f"SOURCE TEXT:\n{text.strip()}")
    else:
        raise HTTPException(400, "Upload a file or paste some text.")

    fig_list = "\n".join(f"- {f['id']} (p.{f['page']}): {f['caption']}" for f in figures) or "(none)"
    parts.append(DIGEST_PROMPT.format(n=n, figures=fig_list))
    resp, errors = None, []
    for model in DIGEST_FALLBACKS:
        try:
            resp = client().models.generate_content(
                model=model,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Deck,
                    temperature=0.4,
                    http_options=types.HttpOptions(timeout=90_000, retry_options=types.HttpRetryOptions(attempts=1)),
                ),
            )
            break
        except HTTPException:
            raise
        except Exception as e:
            errors.append(f"{model}: {str(e)[:120]}")
    if resp is None:  # surface model errors to the UI
        raise HTTPException(502, "Gemini is busy right now, please retry. " + " | ".join(errors))
    deck = resp.parsed if isinstance(resp.parsed, Deck) else Deck.model_validate_json(resp.text)
    out = deck.model_dump()
    by_id = {f["id"]: f for f in figures}
    used = set()
    for c in out["concepts"]:
        f = by_id.get(c.pop("figure_id", "").strip())
        if f and f["id"] not in used:
            used.add(f["id"])
            c["figure"] = {k: f[k] for k in ("page", "caption", "data_url")}
    out["figures_found"] = len(figures)
    out["digest_model"] = model
    return out


# ---------- Tutor (Gemma 4) ----------

def tutor_instructions() -> str:
    """The tutor's rules live in the Agent Skill so the same prompt is reusable elsewhere."""
    try:
        body = SKILL_PATH.read_text()
        return body.split("---", 2)[2].strip() if body.startswith("---") else body
    except OSError:
        return "You are a strict but kind study tutor. Only use the provided source text."


def ask_gemma(prompt: str, thinking: str = "minimal") -> dict:
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
For EACH concept below write one multiple-choice recall question that tests understanding
(not trivia). 4 options, exactly one correct. Wrong options must be plausible misconceptions
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

Return ONLY JSON: {{"score": int 0-100, "verdict": "nailed it" | "almost" | "not yet",
"got_right": [str], "missed": [str], "misconceptions": [str],
"tip": str (one sentence, encouraging, what to say next time)}}"""
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
        "misconceptions": list(out.get("misconceptions") or []),
        "tip": out.get("tip", ""),
        "page": c.page,
        "source_quote": c.source_quote,
        "model": TUTOR_MODEL,
    }


@app.get("/api/health")
def health():
    return {"ok": True, "digest_model": DIGEST_MODEL, "tutor_model": TUTOR_MODEL,
            "key": bool(os.environ.get("GEMINI_API_KEY"))}


STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
