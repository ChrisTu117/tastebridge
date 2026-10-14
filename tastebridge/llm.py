"""Pluggable LLM. Default: offline MockPlanner. Set LLM_PROVIDER=openai_compat with
LLM_BASE_URL (any OpenAI-compatible endpoint),
LLM_MODEL (model id) and LLM_API_KEY to use a real model."""
import json, os, time, urllib.error, urllib.request


class LLM:
    def chat(self, messages, tools=None):
        """Return {'content': str|None, 'tool_calls': [{'id','name','arguments':dict}]}"""
        raise NotImplementedError


def _parse_args(raw):
    """Tolerant tool-argument parsing: dict, JSON string, fenced or trailing junk."""
    if isinstance(raw, dict):
        return raw
    raw = (raw or "").strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:]
    try:
        return json.loads(raw or "{}")
    except ValueError:
        i, j = raw.find("{"), raw.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(raw[i:j + 1])
            except ValueError:
                pass
    return {}


class OpenAICompatLLM(LLM):
    """OpenAI-compatible chat client with retries on 429/5xx/timeouts and an
    ordered model fallback list (LLM_MODEL may be comma-separated)."""
    def __init__(self, base_url, model, api_key, temperature=0.2, timeout=None, retries=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.models = [m.strip() for m in model.split(",") if m.strip()]
        self.model = self.models[0]
        self._key, self.temperature = api_key, temperature
        self.timeout = timeout or float(os.environ.get("LLM_TIMEOUT", "90"))
        self.retries = retries if retries is not None else int(os.environ.get("LLM_RETRIES", "3"))
        self.disable_thinking = os.environ.get("LLM_DISABLE_THINKING", "1") == "1"

    def _post(self, model, messages, tools):
        body = {"model": model, "messages": messages, "temperature": self.temperature}
        if tools:
            body["tools"] = tools
        if self.disable_thinking:
            body["thinking"] = {"type": "disabled"}  # Z.ai/GLM; ignored elsewhere
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]

    def chat(self, messages, tools=None):
        last = None
        for model in self.models:
            for attempt in range(self.retries + 1):
                try:
                    msg = self._post(model, messages, tools)
                    self.model = model
                    calls = [{"id": c.get("id") or f"call{i}", "name": c["function"]["name"],
                              "arguments": _parse_args(c["function"].get("arguments"))}
                             for i, c in enumerate(msg.get("tool_calls") or [])]
                    return {"content": msg.get("content"), "tool_calls": calls}
                except urllib.error.HTTPError as e:
                    last = e
                    if e.code in (400, 401, 403, 404):
                        if e.code in (401, 403):
                            raise
                        break  # bad model id etc. -> next model
                    if e.code != 429 and e.code < 500:
                        raise
                except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
                    last = e
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt * 1.5, 10))
        raise RuntimeError(f"LLM failed on all models {self.models}: {last}")


class MockPlanner(LLM):
    """Deterministic offline planner: emits the canonical research plan as tool
    calls, one city at a time, then hands off to the deterministic synthesizer."""
    def __init__(self, cities, category):
        self.queue = [("competitors", {"city": c, "category": category}) for c in cities] + [("target_audience", {})]
        for c in cities:
            self.queue += [("hotspots", {"city": c}),
                           ("audience_affinity", {"city": c, "domain": "artist"}),
                           ("audience_affinity", {"city": c, "domain": "brand"}),
                           ("audience_affinity", {"city": c, "domain": "movie"}),
                           ("audience_affinity", {"city": c, "domain": "tag"}),
                           ("audience_profile", {"city": c}),
                           ("momentum", {"city": c})]
        self.queue += [("compare_cities", {"city_a": a, "city_b": b}) for a, b in zip(cities, cities[1:])]
        self.n = 0

    def chat(self, messages, tools=None):
        if self.queue:
            name, args = self.queue.pop(0)
            self.n += 1
            return {"content": None, "tool_calls": [{"id": f"m{self.n}", "name": name, "arguments": args}]}
        return {"content": "__SYNTHESIZE__", "tool_calls": []}


def from_env(cities, category):
    if os.environ.get("LLM_PROVIDER") == "openai_compat":
        return OpenAICompatLLM(os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"], os.environ["LLM_API_KEY"])
    return MockPlanner(cities, category)
