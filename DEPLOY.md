# Deploying TasteBridge (free tier)

The app is one stdlib Python process: `python -m tastebridge.server`. It serves the UI at `/` and a streaming NDJSON API at `/api/run`.
Secrets are read only from server environment variables. The browser never gets them, and `tests/test_server.py` checks this.

| Env var | Required | Notes |
|---|---|---|
| `QLOO_API_KEY` | yes | Qloo hackathon key |
| `PORT` | no | default 8080 (Docker image: 7860) |
| `LLM_PROVIDER` | no | unset = offline mock planner; `openai_compat` = real LLM |
| `LLM_BASE_URL` | with openai_compat | any OpenAI-compatible base URL, e.g. `https://api.z.ai/api/paas/v4/` |
| `LLM_MODEL` | with openai_compat | model id(s), comma-separated fallback, e.g. `glm-4.7-flash,glm-4.5-flash` |
| `LLM_API_KEY` | with openai_compat | provider API key (set as a Secret) |
| `QLOO_CACHE` | no | response cache dir (default `.cache/qloo`) |

## Option A (recommended): Hugging Face Spaces, Docker SDK
Free CPU, no credit card, public URL `https://<user>-tastebridge.hf.space`.
1. Create a Space with SDK = Docker, then push this repo to it. The included `Dockerfile` listens on 7860.
2. Under Settings → Variables and secrets, add `QLOO_API_KEY` as a **Secret** (plus the `LLM_*` variables later).
3. Free Spaces sleep after inactivity. Open the Space once before the judges look at it.

## Option B: Render free web service
Choose New → Web Service from GitHub with runtime Docker, and set the env vars in the dashboard.
Free instances spin down when idle, so the first request after that takes about 50 s.

## Why not Cloudflare Workers directly?
Workers run Python on Pyodide, which has no `urllib` sockets. It would need a fetch-based rewrite.
Workers also cap CPU time per request, and a run makes about 20 sequential Qloo calls.
If a Cloudflare URL is wanted, put a Worker or Pages proxy in front of option A or B.

## Local
```
QLOO_API_KEY=... python3 -m tastebridge.server   # http://localhost:8080
python3 -m unittest discover -s tests -t .
```

## Render (recommended, free, no card)
1. Sign up at https://render.com with GitHub (no card needed).
2. New > Blueprint > pick this repo (render.yaml, Docker, plan free).
3. Paste secrets when prompted: `LLM_API_KEY`, `QLOO_API_KEY`.
4. URL: https://tastebridge-XXXX.onrender.com. Free tier sleeps after 15 min idle (~1 min cold start); open it once before the demo.
