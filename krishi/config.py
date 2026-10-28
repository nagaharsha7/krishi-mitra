"""Central configuration. The API key is read ONLY from the .env file / environment."""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# One model for chat AND Crop Doctor (text + image).
# NOTE: this default was NOT verified from my side (no access to Google's API when this was written).
# Run `python check_models.py --save` once: it lists real Gemma models, tests image input,
# and writes the correct GEMMA_MODEL into your .env (which overrides this default).
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-31b-it").strip()

DEMO_MODE = os.getenv("DEMO_MODE", "false").strip().lower() in ("1", "true", "yes")

MAX_RETRIES = 2            # retries for HTTP 429 / 503
RETRY_WAIT_SECONDS = 2     # wait grows: 2s, 4s
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
