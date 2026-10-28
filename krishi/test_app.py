import io, json, time
from unittest import mock
import config, gemini_client as gc
from google.genai import errors
import app as appmod
c = appmod.app.test_client()
config.RETRY_WAIT_SECONDS = 0.01
PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360000002000001e221bc330000000049454e44ae426082")
class R: text = "తెలుగు సమాధానం"
def api_err(code): return errors.APIError(code, {"error": {"message": f"x {code}", "status": "X"}})

def run(name, ok): print(("PASS " if ok else "FAIL ") + name); assert ok

# Scheme Finder: no AI, no key, no network
config.GEMINI_API_KEY = ""
d = c.post("/api/schemes", json={"state": "Telangana", "farmer_type": "owner", "acres": 3}).get_json()
run("schemes works w/o key", d["ok"] and len(d["schemes"]) >= 5)
d = c.post("/api/schemes", json={"acres": 50, "farmer_type": "owner", "state": "Karnataka"}).get_json()
run("schemes filter (acres/state)", all(s["id"] not in ("pmksy", "rythu_bharosa") for s in d["schemes"]))
run("schemes bad acres -> 400", c.post("/api/schemes", json={"acres": "abc"}).status_code == 400)

# missing key
r = c.post("/api/chat", json={"message": "వరి"}); run("chat missing key friendly", r.status_code == 503 and "GEMINI_API_KEY" in r.get_json()["message"])
r = c.get("/api/health"); run("health missing key", r.status_code == 503 and r.get_json()["model"] == config.GEMMA_MODEL)

# success with mock
config.GEMINI_API_KEY = "fake"
with mock.patch.object(gc, "_client") as m:
    m.return_value.models.generate_content.return_value = R()
    r = c.post("/api/chat", json={"message": "వరికి నీరు?", "history": [{"role": "user", "text": "hi"}, {"role": "model", "text": "hello"}]})
    run("chat ok", r.get_json()["reply"] == R.text)
    sent = m.return_value.models.generate_content.call_args.kwargs
    run("model passed + tips grounding in prompt", sent["model"] == config.GEMMA_MODEL and "అగ్గితెగులు" in str(sent["contents"][0]))
    r = c.post("/api/crop-doctor", data={"image": (io.BytesIO(PNG), "a.png", "image/png")}, content_type="multipart/form-data")
    run("crop doctor ok", r.get_json()["ok"])
    part = m.return_value.models.generate_content.call_args.kwargs["contents"][0]
    run("image sent as inline bytes w/ mime", part.inline_data.mime_type == "image/png" and part.inline_data.data == PNG)
    run("health ok shows model", c.get("/api/health").get_json() == {"status": "ok", "model": config.GEMMA_MODEL, "demo_mode": False, "error": None, "message": None})
    r = c.post("/api/crop-doctor", data={"image": (io.BytesIO(b"x"), "a.gif", "image/gif")}, content_type="multipart/form-data")
    run("bad mime rejected", r.status_code == 400)
    r = c.post("/api/crop-doctor", data={}, content_type="multipart/form-data"); run("no image -> 400", r.status_code == 400)

# 429 retries then friendly; 429 then success; 503; wrong key; 404
with mock.patch.object(gc, "_client") as m:
    g = m.return_value.models.generate_content
    g.side_effect = api_err(429)
    r = c.post("/api/chat", json={"message": "x"}); run("429 -> quota msg", r.status_code == 429 and "quota" in r.get_json()["message"])
    run("429 retried exactly 2 times (3 calls)", g.call_count == 3)
    g.reset_mock(); g.side_effect = [api_err(503), R()]
    run("503 then success", c.post("/api/chat", json={"message": "x"}).get_json()["ok"] and g.call_count == 2)
    g.reset_mock(); g.side_effect = api_err(403)
    r = c.post("/api/chat", json={"message": "x"}); run("403 -> invalid key, no retry", r.status_code == 401 and g.call_count == 1)
    g.side_effect = RuntimeError("boom"); run("generic exception -> no crash", c.post("/api/chat", json={"message": "x"}).status_code == 503)

# REAL network failure (sandbox has no access to Google) with fake key
config.GEMINI_API_KEY = "wrong"; config.RETRY_WAIT_SECONDS = 0.01
r = c.post("/api/chat", json={"message": "hi"}); j = r.get_json()
run("real no-internet -> friendly bilingual", r.status_code == 503 and "ఇంటర్నెట్" in j["message"] and "internet" in j["message"])
run("health no-internet", c.get("/api/health").status_code == 503)

# demo mode
config.DEMO_MODE = True; config.GEMINI_API_KEY = ""
run("demo chat", c.post("/api/chat", json={"message": "x"}).get_json()["demo"])
r = c.post("/api/crop-doctor", data={"image": (io.BytesIO(PNG), "a.png", "image/png")}, content_type="multipart/form-data")
run("demo crop", r.get_json()["demo"])
run("demo health", c.get("/api/health").get_json()["status"] == "demo")
config.DEMO_MODE = False
run("index page", c.get("/").status_code == 200)
print("ALL TESTS PASSED")
