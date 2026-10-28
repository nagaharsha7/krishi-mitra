"""Lists every Gemma model available to your key and tests which ones accept images.

Usage:
    python check_models.py          # list + test
    python check_models.py --save   # also write the best choice to .env as GEMMA_MODEL
"""
import base64
import re
import sys
from pathlib import Path

from google import genai
from google.genai import types

import config

# 1x1 PNG used to test image input
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def short(name: str) -> str:
    return name.split("/", 1)[1] if name.startswith("models/") else name


def supports_image(client, model: str):
    try:
        r = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=TINY_PNG, mime_type="image/png"),
                "Reply with the single word OK.",
            ],
        )
        return True, (r.text or "").strip()[:40]
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:120]


def main():
    if not config.GEMINI_API_KEY:
        print("GEMINI_API_KEY is missing. Create a .env file (see .env.example).")
        sys.exit(1)
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    try:
        models = [m for m in client.models.list() if "gemma" in m.name.lower()]
    except Exception as e:  # noqa: BLE001
        print("Could not list models:", e)
        sys.exit(1)

    if not models:
        print("No Gemma models are visible to this key.")
        sys.exit(1)

    print(f"{'MODEL':40} generateContent")
    for m in sorted(models, key=lambda x: x.name):
        gc = "generateContent" in (m.supported_actions or [])
        print(f"{short(m.name):40} {'YES' if gc else 'no'}")

    cands = [
        short(m.name) for m in models
        if "generateContent" in (m.supported_actions or [])
        and re.search(r"gemma-?4", m.name.lower())
        and re.search(r"(26b|31b)", m.name.lower())
    ]
    # prefer 31B, then 26B
    cands.sort(key=lambda n: (0 if "31b" in n.lower() else 1, n))

    if not cands:
        print("\nNo Gemma 4 26B/31B model with generateContent found.")
        print("Tell me which model to use instead.")
        sys.exit(2)

    print("\nTesting image input on Gemma 4 candidates...")
    best = None
    for n in cands:
        ok, info = supports_image(client, n)
        print(f"  {n:38} image: {'YES' if ok else 'NO'}  ({info})")
        if ok and not best:
            best = n

    if best:
        print(f"\nBEST CHOICE (text + image): {best}")
        if "--save" in sys.argv:
            env = Path(".env")
            lines = env.read_text().splitlines() if env.exists() else []
            lines = [l for l in lines if not l.startswith("GEMMA_MODEL=")]
            lines.append(f"GEMMA_MODEL={best}")
            env.write_text("\n".join(lines) + "\n")
            print("Saved GEMMA_MODEL to .env")
        else:
            print(f"Set GEMMA_MODEL={best} in .env (or run with --save).")
    else:
        print("\nNo Gemma 4 model accepted images. Chat can use Gemma; "
              "Crop Doctor needs a different model. Tell me which one to use.")


if __name__ == "__main__":
    main()
