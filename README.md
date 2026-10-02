# 🌀 Unscroll

**Your assigned reading, as a feed you can actually finish and remember.**

Unscroll turns a PDF reading, a photo of a textbook page, or pasted notes into a short-form vertical feed of narrated "reels", one per key concept. Active-recall checks are built into the feed: quiz cards, "teach it back" explanations graded by AI against the source, and a spaced-review queue that brings missed concepts back the next day.

## The problem
At San Francisco State University, most students commute, many are first-generation, and many work or are learning in their second language. Assigned readings are long and get skipped. Short-form video holds attention but teaches nothing that sticks. Unscroll keeps the format students already use and puts real learning science behind it: retrieval practice, the Feynman technique, and spaced repetition.

**Who benefits at SFSU**
- **Students**, especially commuters (listen on Muni/BART), English-language learners (plain-language scripts, grading that ignores grammar), and students with ADHD or dyslexia (captions plus audio, short chunks)
- **Faculty**, whose readings actually get done, with comprehension checks they didn't have to write
- **DPRC** (Disability Programs & Resource Center), as an accessible alternative format for course readings

## How it works
```
PDF / image / text
      │
      ▼
PyMuPDF ── crops the reading's REAL captioned figures (Figure 8.x, tables, charts)
      │
      ▼
Gemini Flash ── multimodal read → structured JSON (concepts, 30-s scripts, key points,
      │          verbatim quote + page, and which source figure illustrates each concept)
      │
      ▼
Swipe feed (scroll-snap) ── narrated reels + original figures + karaoke captions + page citations
      │
      ├─► Gemma 4 /api/quiz       → recall MCQs with misconception distractors
      ├─► Gemma 4 /api/teachback  → grades the student's typed or spoken explanation ONLY against the source
      └─► Review queue            → missed concepts are due again tomorrow
```

## Models (open-weight + Gemini)
| Role | Model | Access | License / terms |
|---|---|---|---|
| Digest (multimodal PDF → concepts) | `gemini-3.8-flash` | Gemini API | [Gemini API terms](https://ai.google.dev/gemini-api/terms) |
| Tutor (quiz + teach-back grading) | **Gemma 4** `gemma-4-26b-a4b-it` (open-weight) | Gemini API | [Gemma license / terms](https://ai.google.dev/gemma/terms) |

Why we split it this way: Gemini handles the heavy multimodal reading once per document. Gemma 4 is the open-weight tutor that runs on every student interaction. Because Gemma's weights are open, the tutor can run **on-device or on SFSU infrastructure** (for example through Ollama), so students' answers never have to leave campus. To swap models, set `TUTOR_MODEL` / `DIGEST_MODEL`.

Gemma integration is in [`app.py`](app.py) (`ask_gemma`, `/api/quiz`, `/api/teachback`).

## Agent Skill
The tutor's behavior is packaged as an [Agent Skill](https://agentskills.io/) at [`skills/unscroll-tutor/SKILL.md`](skills/unscroll-tutor/SKILL.md). The app loads it as Gemma's system instruction at runtime, and any skills-compatible agent can reuse it to quiz students on a source and grade their explanations.

## Google tools used
- **Gemini API** (Gemini 3.8 Flash + Gemma 4), with keys from **Google AI Studio**
- **Cloud Run**, which hosts the app and is paid for with the hackathon Google Cloud credits

## Run locally
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=your_key
.venv/bin/uvicorn app:app --reload --port 8080
```
Open http://localhost:8080 and click **Use sample reading**.

## Deploy to Cloud Run
```bash
gcloud run deploy unscroll --source . --region us-west1 --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key
```

## Responsible AI
| Risk | What we do |
|---|---|
| Fake or misleading visuals | We never generate images. Visuals are only the reading's own figures, cropped unedited and labeled "FROM YOUR READING · p.N". Narration and scripts are labeled as AI-generated. |
| Hallucination | Every reel shows a verbatim quote and page number. Prompts forbid outside facts. Teach-back is graded only against the source. |
| Privacy | Uploads are processed in memory and never stored. There are no accounts, and the review queue lives in the student's own browser. The Gemma tutor can be self-hosted. |
| Bias / language | The grader is told to ignore grammar, spelling, and transcription errors so ESL students aren't penalized. Scripts are written at a 9th-grade reading level. |
| Accessibility | Captions plus narration, large text, keyboard-scrollable feed, reduced-motion support, and voice input for teach-back. |
| Academic integrity | It's a study tool, not a summary generator. The core loop forces the student to produce recall. |
| Copyright | Only processes material the user uploads, and output stays private to that user. |

## Path to a pilot at SFSU
1. Pilot with one large gen-ed course (GE Area D) and the DPRC for one semester. Measure reading completion and quiz scores against a control section.
2. Integrate with Canvas so instructors can attach an Unscroll feed to a reading assignment.
3. Self-host Gemma 4 on campus or in SFSU's GCP tenant for FERPA-friendly grading. Add Firebase Auth with SFSU SSO and Firestore for review across devices.
4. Future: Gemini TTS podcast mode for commutes, generated diagrams per reel, multilingual feeds.

## License
[MIT](LICENSE)
