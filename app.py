"""Unscroll: turn assigned readings into a swipeable learn-feed.

Gemini 2.5 Flash digests the reading (multimodal PDF -> concept reels).
Gemma 4 (open-weight, via the Gemini API) is the tutor: quizzes + teach-back grading.
"""

import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel

DIGEST_MODEL = os.environ.get("DIGEST_MODEL", "gemini-2.5-flash")
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


class Deck(BaseModel):
    deck_title: str
    subject: str
    concepts: list[Concept]


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

deck_title: a catchy title for the whole reading. subject: the academic subject.
"""


@app.post("/api/ingest")
async def ingest(file: UploadFile | None = File(None), text: str = Form(""), n: int = Form(8)):
    n = max(3, min(n, 12))
    parts: list = []
    if file is not None and file.filename:
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (20 MB max).")
        mime = file.content_type or "application/pdf"
        if mime == "text/plain":
            parts.append(data.decode("utf-8", errors="ignore"))
        else:
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    elif text.strip():
        parts.append(f"SOURCE TEXT:\n{text.strip()}")
    else:
        raise HTTPException(400, "Upload a file or paste some text.")

    parts.append(DIGEST_PROMPT.format(n=n))
    try:
        resp = client().models.generate_content(
            model=DIGEST_MODEL,
            contents=parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Deck,
                temperature=0.4,
            ),
        )
    except HTTPException:
        raise
    except Exception as e:  # surface model errors to the UI
        raise HTTPException(502, f"Gemini error: {e}")
    deck = resp.parsed if isinstance(resp.parsed, Deck) else Deck.model_validate_json(resp.text)
    return deck.model_dump()


# ---------- Tutor (Gemma 4) ----------

def tutor_instructions() -> str:
    """The tutor's rules live in the Agent Skill so the same prompt is reusable elsewhere."""
    try:
        body = SKILL_PATH.read_text()
        return body.split("---", 2)[2].strip() if body.startswith("---") else body
    except OSError:
        return "You are a strict but kind study tutor. Only use the provided source text."


def ask_gemma(prompt: str) -> dict:
    try:
        resp = client().models.generate_content(
            model=TUTOR_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=tutor_instructions(), temperature=0.3
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
(not trivia). 4 options, exactly one correct. Wrong options must be plausible misconceptions.

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
    out = ask_gemma(prompt)
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
