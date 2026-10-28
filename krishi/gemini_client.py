"""Thin, crash-proof wrapper around the google-genai SDK (Gemma via Gemini Developer API)."""
import time

from google import genai
from google.genai import types

import config
import demo_answers

MESSAGES = {
    "no_key": ("API కీ కనుగొనబడలేదు. .env ఫైల్‌లో GEMINI_API_KEY ను చేర్చండి.",
               "API key missing. Add GEMINI_API_KEY to your .env file."),
    "bad_key": ("API కీ చెల్లదు. దయచేసి సరైన కీని .env ఫైల్‌లో ఉంచండి.",
                "The API key is invalid. Please put a valid key in your .env file."),
    "quota": ("ఉచిత పరిమితి అయిపోయింది. కొంత సేపటి తర్వాత మళ్లీ ప్రయత్నించండి.",
              "Free quota exceeded. Please try again in a little while."),
    "busy": ("సర్వర్ బిజీగా ఉంది. కొంత సేపటి తర్వాత మళ్లీ ప్రయత్నించండి.",
             "The AI server is busy. Please try again shortly."),
    "network": ("ఇంటర్నెట్ కనెక్షన్ లేదు. కనెక్షన్ చూసి మళ్లీ ప్రయత్నించండి.",
                "No internet connection. Check your network and try again."),
    "model": ("ఈ మోడల్ అందుబాటులో లేదు. check_models.py నడపండి.",
              "Model not available. Run check_models.py and set GEMMA_MODEL."),
    "unknown": ("ఏదో తప్పు జరిగింది. మళ్లీ ప్రయత్నించండి.",
                "Something went wrong. Please try again."),
}
STATUS = {"no_key": 503, "bad_key": 401, "quota": 429, "busy": 503,
          "network": 503, "model": 502, "unknown": 500}


def friendly(kind: str) -> str:
    te, en = MESSAGES.get(kind, MESSAGES["unknown"])
    return f"{te}\n{en}"


def _classify(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        if code == 429:
            return "quota"
        if code in (500, 502, 503, 504):
            return "busy"
        if code in (401, 403):
            return "bad_key"
        if code == 404:
            return "model"
        if code == 400:
            text = str(exc).lower()
            return "bad_key" if "api key" in text or "api_key" in text else "unknown"
        return "unknown"
    # no HTTP code -> connection-level problem (DNS, timeout, refused...)
    return "network"


def _client():
    return genai.Client(api_key=config.GEMINI_API_KEY)


def generate(contents, max_output_tokens=1024):
    """Returns {"ok": bool, "text": str, "kind": str|None, "message": str|None}. Never raises."""
    if config.DEMO_MODE:
        return {"ok": True, "text": None, "kind": "demo", "message": None}
    if not config.GEMINI_API_KEY:
        return _err("no_key")
    try:
        client = _client()
    except Exception:  # noqa: BLE001
        return _err("unknown")

    attempt = 0
    while True:
        try:
            resp = client.models.generate_content(
                model=config.GEMMA_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens, temperature=0.4),
            )
            text = (resp.text or "").strip()
            if not text:
                return _err("unknown")
            return {"ok": True, "text": text, "kind": None, "message": None}
        except Exception as e:  # noqa: BLE001
            kind = _classify(e)
            if kind in ("quota", "busy") and attempt < config.MAX_RETRIES:
                attempt += 1
                time.sleep(config.RETRY_WAIT_SECONDS * attempt)
                continue
            return _err(kind)


def _err(kind):
    return {"ok": False, "text": None, "kind": kind, "message": friendly(kind)}


def build_text(system_prompt, history, user_msg):
    """Gemma may reject a separate system role, so the system prompt is placed in the first user turn."""
    contents = []
    first = True
    for h in history[-10:]:
        role = "model" if h.get("role") == "model" else "user"
        txt = str(h.get("text", ""))[:2000]
        if not txt:
            continue
        if first and role == "user":
            txt = system_prompt + "\n\n" + txt
            first = False
        contents.append(types.Content(role=role, parts=[types.Part(text=txt)]))
    final = user_msg if not first or contents else system_prompt + "\n\nరైతు ప్రశ్న: " + user_msg
    contents.append(types.Content(role="user", parts=[types.Part(text=final)]))
    return contents


def ping():
    """Used by /api/health. Returns dict with status."""
    if config.DEMO_MODE:
        return {"status": "demo", "model": config.GEMMA_MODEL, "demo_mode": True}
    r = generate("Reply with the single word OK.", max_output_tokens=16)
    return {
        "status": "ok" if r["ok"] else "error",
        "model": config.GEMMA_MODEL,
        "demo_mode": False,
        "error": r["kind"],
        "message": r["message"],
    }
