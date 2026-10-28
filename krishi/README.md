# 🌾 కృషి మిత్ర (Krishi Mitra)

Telugu farming assistant: **Chat**, **Crop Doctor** (photo), **Scheme Finder** (offline).
AI = Google **Gemma 4** via Google AI Studio (Gemini Developer API), free key. No Ollama, no Groq.

## 1. Get a free API key
1. Open https://aistudio.google.com and sign in.
2. Click **Get API key** → **Create API key**. Copy it.

## 2. Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```
Open `.env` and set `GEMINI_API_KEY=your_key`. Never commit `.env` (it is in `.gitignore`).

## 3. Pick the right Gemma 4 model (do not guess the name)
```bash
python check_models.py --save
```
This lists every Gemma model for your key (with `generateContent` support), tests image input on the
Gemma 4 26B/31B models, and saves the best one as `GEMMA_MODEL` in `.env`.
If no Gemma 4 model accepts images, Chat still works but Crop Doctor needs a different model.

## 4. Run
```bash
python app.py
```
Open http://127.0.0.1:5000 · Health check: http://127.0.0.1:5000/api/health

## Demo mode
`DEMO_MODE=true` in `.env` returns pre-written Telugu answers with no API calls (default: false).

## Errors
Missing key, invalid key, quota (429), busy server (503) and no internet all return a friendly
Telugu + English message. 429/503 are retried up to 2 times.

## Scheme Finder
Works with no AI and no internet (data in `data/schemes.json`). Verify details on official sites.

## Privacy note
On the **free tier**, Google may use your prompts and images to improve its products (including training).
**Do not send private or personal data** (names, Aadhaar, phone numbers, bank details).
