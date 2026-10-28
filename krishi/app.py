import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from google.genai import types

import config
import demo_answers
import gemini_client as gc

BASE = Path(__file__).parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_IMAGE_BYTES + 512 * 1024

TIPS = json.loads((BASE / "data" / "farming_tips.json").read_text(encoding="utf-8"))
SCHEMES = json.loads((BASE / "data" / "schemes.json").read_text(encoding="utf-8"))

SYSTEM_PROMPT = (
    "మీరు 'కృషి మిత్ర' – తెలుగు రైతులకు సహాయపడే వ్యవసాయ సహాయకుడు. "
    "ఎల్లప్పుడూ సరళమైన తెలుగులో, చిన్న వాక్యాల్లో సమాధానం ఇవ్వండి. "
    "క్రింది వ్యవసాయ సూచనలను ఆధారంగా తీసుకోండి; వాటిలో లేని విషయాలలో జాగ్రత్తగా, సాధారణ సలహా మాత్రమే ఇవ్వండి. "
    "మందుల మోతాదు గురించి తెలియకపోతే ఊహించవద్దు; స్థానిక వ్యవసాయ అధికారిని సంప్రదించమని చెప్పండి. "
    "వ్యవసాయేతర ప్రశ్నలకు మర్యాదగా తిరస్కరించండి.\n\nవ్యవసాయ సూచనలు:\n"
    + "\n".join(f"- {t['topic']}: {t['tip']}" for t in TIPS)
)

CROP_PROMPT = (
    SYSTEM_PROMPT
    + "\n\nఈ ఫోటోలో ఉన్న పంటను పరిశీలించండి. తెలుగులో ఈ క్రమంలో చెప్పండి: "
      "1) పంట 2) సమస్య/తెగులు (అంచనా) 3) చికిత్స 4) నివారణ. "
      "ఖచ్చితంగా తెలియకపోతే అది అంచనా మాత్రమే అని చెప్పి, స్థానిక వ్యవసాయ అధికారిని సంప్రదించమని సూచించండి."
)


def fail(r):
    return jsonify(ok=False, message=r["message"], kind=r["kind"]), gc.STATUS.get(r["kind"], 500)


@app.get("/")
def index():
    return render_template("index.html", model=config.GEMMA_MODEL, demo=config.DEMO_MODE)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    msg = str(data.get("message", "")).strip()[:2000]
    if not msg:
        return jsonify(ok=False, message="దయచేసి ప్రశ్న రాయండి.\nPlease type a question."), 400
    if config.DEMO_MODE:
        return jsonify(ok=True, reply=demo_answers.CHAT, demo=True)
    history = data.get("history") if isinstance(data.get("history"), list) else []
    contents = gc.build_text(SYSTEM_PROMPT, [h for h in history if isinstance(h, dict)], msg)
    r = gc.generate(contents)
    return jsonify(ok=True, reply=r["text"]) if r["ok"] else fail(r)


@app.post("/api/crop-doctor")
def crop_doctor():
    f = request.files.get("image")
    if not f:
        return jsonify(ok=False, message="దయచేసి పంట ఫోటో ఎంచుకోండి.\nPlease choose a crop photo."), 400
    mime = (f.mimetype or "").lower()
    if mime not in config.ALLOWED_MIME:
        return jsonify(ok=False, message="JPG, PNG లేదా WEBP ఫోటో మాత్రమే.\nOnly JPG, PNG or WEBP images."), 400
    data = f.read()
    if not data or len(data) > config.MAX_IMAGE_BYTES:
        return jsonify(ok=False, message="ఫోటో పరిమాణం 5MB లోపు ఉండాలి.\nImage must be under 5MB."), 400
    if config.DEMO_MODE:
        return jsonify(ok=True, reply=demo_answers.CROP, demo=True)
    contents = [types.Part.from_bytes(data=data, mime_type=mime), CROP_PROMPT]
    r = gc.generate(contents, max_output_tokens=1200)
    return jsonify(ok=True, reply=r["text"]) if r["ok"] else fail(r)


@app.post("/api/schemes")
def schemes():
    """Pure local filtering: no AI, no internet."""
    d = request.get_json(silent=True) or {}
    state = str(d.get("state", "")).strip()
    ftype = str(d.get("farmer_type", "")).strip()
    try:
        acres = float(d["acres"]) if str(d.get("acres", "")).strip() != "" else None
    except (ValueError, TypeError):
        return jsonify(ok=False, message="భూమి విస్తీర్ణం సంఖ్యగా ఇవ్వండి.\nEnter land size as a number."), 400
    out = []
    for s in SCHEMES:
        if s["states"] and state and state not in s["states"]:
            continue
        if ftype and ftype not in s["farmer_types"]:
            continue
        if acres is not None and s["max_land_acres"] is not None and acres > s["max_land_acres"]:
            continue
        out.append(s)
    return jsonify(ok=True, schemes=out, note="వివరాలను అధికారిక వెబ్‌సైట్‌లో నిర్ధారించుకోండి. / Please confirm details on the official site.")


@app.get("/api/health")
def health():
    h = gc.ping()
    return jsonify(h), (200 if h["status"] in ("ok", "demo") else 503)


@app.errorhandler(413)
def too_big(_e):
    return jsonify(ok=False, message="ఫైల్ చాలా పెద్దది.\nFile too large."), 413


@app.errorhandler(Exception)
def any_error(_e):
    return jsonify(ok=False, message=gc.friendly("unknown"), kind="unknown"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
