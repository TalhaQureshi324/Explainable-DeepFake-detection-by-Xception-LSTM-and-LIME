# Visual Review Guide — run the full application locally

Everything needed to review the homepage, upload screen, live processing,
results, interactive report and the downloadable PDF on your own machine.

## 1. Prerequisites (already true on the dev machine)

- Python 3.11 at `C:\Program Files\Python311\python.exe` with the repo deps
  (`pip install -r requirements.txt -r backend/requirements.txt`)
- Checkpoints in `Fyp_Git_aligned/models_store/` (V1 + V2)
- Node.js 18+ and `cd frontend && npm install`

## 2. Start the two servers (two terminals, repo root)

```bash
# Terminal 1 — inference backend
"C:\Program Files\Python311\python.exe" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend (proxy target read at RUNTIME from INFERENCE_API_URL)
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** — use the sun/moon button (dark/light) and the
اردو button (English/Urdu RTL) in the header on every page.

## 3. What to review, in order

| Page | URL | What to look at |
|------|-----|-----------------|
| Homepage | `/` | Hero, research-prototype disclaimer, 4-step architecture cards, limitations notice; try dark mode + Urdu (RTL flips the whole layout) |
| Upload | `/analyze` | Drag-drop a video or image, preview, model V1/V2 picker, LIME toggle, size/type errors |
| Processing | `/analysis/<id>` (automatic after Analyze) | Real backend states only — no fake percentages |
| Results | same page, when COMPLETED | Verdict + uncalibrated-score notice, score vs threshold bar, model/checkpoint info, warnings, window timeline (videos), processed face |
| Report | click "Open detailed report" | Executive overview, prediction summary, media info (size/sha/fps/frames), timeline chart, visual evidence, technical + preprocessing details, interpretation, limitations, **Download PDF** |
| PDF | "Download PDF" button | Server-rendered from the same report data; opens in any PDF reader |

## 4. Fastest demo media

Any small file from `F:\Computer Vision\FYP\data\val\…` (FAKE videos are the
interesting case). A complete analysis takes ~1–3 minutes per video on CPU
(LIME adds ~1–3 minutes; `DEEPFAKE_LIME_SAMPLES=20` shortens it for demos).

## 5. Notes

- Phase 1/3 keep jobs in memory: restarting the backend clears previous
  analyses (the UI explains this if you reopen an old link).
- Uploaded media is deleted as soon as the analysis finishes; only the face
  preview / LIME artifacts and the cached PDF remain in `backend/.artifacts/`.
