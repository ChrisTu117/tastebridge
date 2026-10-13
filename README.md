---
title: TasteBridge
emoji: 🍵
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
license: mit
pinned: false
---

# TasteBridge

Cultural go-to-market agent for Chinese consumer brands entering overseas cities.
It's grounded in the [Qloo](https://qloo.com) taste graph, and its planner runs on any OpenAI-compatible LLM endpoint.

```
python3 -m tastebridge --cities "Los Angeles,London,Singapore" --json out/report.json
python3 -m unittest discover -s tests -t .
QLOO_API_KEY=... python3 -m tastebridge.server   # web UI on :8080
```

See DEPLOY.md for free hosting.
```
```

Env: `QLOO_API_KEY` (or `~/.secrets/qloo.env`). By default the CLI runs the offline mock planner.
To use a real LLM, point it at any OpenAI-compatible endpoint:
`LLM_PROVIDER=openai_compat LLM_BASE_URL=<base url ending in /v1 or equivalent> LLM_MODEL=<model id> LLM_API_KEY=...`
(e.g. Z.ai `https://api.z.ai/api/paas/v4/` with `glm-4.7-flash`, or Groq `https://api.groq.com/openai/v1`).

License: MIT
