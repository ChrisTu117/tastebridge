"""Stdlib HTTP server. The Qloo/LLM keys live only in server env; the browser
talks to /api/* and never sees them."""
import json, os, sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from .qloo import Qloo, CITY_COORDS
from .llm import from_env
from .agent import Agent, render

STATIC = Path(__file__).parent / "static"
MAX_CITIES = 4


def validate(body):
    brand = str(body.get("brand", "")).strip()[:80] or "Unnamed brand"
    category = str(body.get("category", "bubble tea")).strip()[:40] or "bubble tea"
    cities = [str(c).strip()[:40] for c in body.get("cities", []) if str(c).strip()]
    if not 1 <= len(cities) <= MAX_CITIES:
        raise ValueError(f"give 1-{MAX_CITIES} cities")
    return {"brand": brand, "category": category, "cities": cities}


def make_agent(cities, category):
    return Agent(Qloo(cache_dir=os.environ.get("QLOO_CACHE", ".cache/qloo")), from_env(cities, category))


class H(BaseHTTPRequestHandler):
    agent_factory = staticmethod(make_agent)

    def log_message(self, fmt, *a):  # no query strings / headers in logs
        sys.stderr.write("%s %s\n" % (self.command, self.path.split("?")[0]))

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/health":
            return self._send(200, {"ok": True, "llm": os.environ.get("LLM_PROVIDER", "mock"), "cities": sorted(CITY_COORDS)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/run":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 10000: raise ValueError("body too large")
            brief = validate(json.loads(self.rfile.read(n) or b"{}"))
        except Exception as e:
            return self._send(400, {"error": str(e)})
        # Stream NDJSON events so the UI can render the agent trace live.
        self.send_response(200); self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache"); self.end_headers()
        def emit(ev):
            self.wfile.write((json.dumps(ev, ensure_ascii=False) + "\n").encode()); self.wfile.flush()
        emit({"type": "start", "brief": brief, "llm": os.environ.get("LLM_PROVIDER", "mock")})
        try:
            rep = self.agent_factory(brief["cities"], brief["category"]).run(brief, on_event=emit)
            for r in rep["ranking"]:
                r["coords"] = CITY_COORDS.get(r["city"].lower())
            emit({"type": "report", "report": {k: rep.get(k) for k in ("brief", "weights", "ranking", "errors", "llm_narrative")},
                  "evidence": [{k: e[k] for k in ("id", "kind", "city", "call")} | {"count": len(e["data"])} for e in rep["evidence"]],
                  "markdown": render(rep)})
        except Exception as e:
            emit({"type": "error", "message": str(e)[:200]})


def main():
    port = int(os.environ.get("PORT", 8080))
    print(f"TasteBridge on http://0.0.0.0:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
