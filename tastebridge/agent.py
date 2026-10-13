"""Agent loop: LLM plans tool calls -> Qloo tools -> evidence store -> synthesis."""
import json
from statistics import mean
from .qloo import Qloo, is_chinese_brand, is_noise

DEFAULT_SIGNAL_TAGS = ["urn:tag:genre:place:restaurant:bubble_tea"]
DEFAULT_KEYWORDS = ["bubble tea", "milk tea", "boba", "tea house", "HEYTEA", "Chagee", "Mixue", "Gong Cha", "Nayuki", "Molly Tea"]
# Target audience for a new-style Chinese tea brand: under-35, female-leaning.
DEFAULT_TARGET = {"age": {"24_and_younger": 1, "25_to_29": 1, "30_to_34": 0.5, "35_to_44": -0.3, "45_to_54": -0.7, "55_and_older": -1},
                  "gender": {"female": 0.5, "male": -0.5}}

TOOLS = [
    {"type": "function", "function": {"name": "competitors", "description": "Geo-filtered multi-keyword scan of competitor venues in a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "keywords": {"type": "array", "items": {"type": "string"}}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "audience_affinity", "description": "What the category's audience in a city also likes, in one domain (artist|brand|movie|tv_show|tag).",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "domain": {"type": "string"}}, "required": ["city", "domain"]}}},
    {"type": "function", "function": {"name": "audience_profile", "description": "Age/gender skew of the city's category audience (via demographics of its top affinity artists/films). Call after audience_affinity.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "hotspots", "description": "Geo hotspots where the city's competitor audience concentrates (heatmap). Call after competitors.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_entity", "description": "Look up any Qloo entity by name.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "type": {"type": "string"}}, "required": ["query"]}}},
]

SYSTEM = ("You are TasteBridge, a go-to-market analyst for Chinese consumer brands entering overseas cities. "
          "For every candidate city call competitors, hotspots, audience_affinity (artist, brand, movie, tag) and audience_profile, "
          "then write the entry plan. Every claim must cite evidence ids like [E3]. Never invent data.")


def fit_score(profile, target=DEFAULT_TARGET):
    """Cosine-like agreement between audience skew and target skew, mapped to 0..1."""
    if not profile:
        return None
    num = den = 0.0
    for dim in ("age", "gender"):
        for k, w in target[dim].items():
            v = (profile.get(dim) or {}).get(k)
            if v is not None:
                num += v * w; den += abs(w)
    return round(max(0.0, min(1.0, 0.5 + num / den)), 3) if den else None


def merge_profiles(ps):
    out = {}
    for dim in ("age", "gender"):
        keys = {k for p in ps for k in (p.get(dim) or {})}
        out[dim] = {k: round(mean([p[dim][k] for p in ps if k in (p.get(dim) or {})]), 3) for k in keys}
    return out


class Agent:
    def __init__(self, qloo: Qloo, llm, signal_tags=None, keywords=None, max_steps=60):
        self.q, self.llm = qloo, llm
        self.tags, self.keywords, self.max_steps = signal_tags or DEFAULT_SIGNAL_TAGS, keywords or DEFAULT_KEYWORDS, max_steps
        self.evidence = []
        self.errors = []

    def _add(self, kind, city, data, call):
        eid = f"E{len(self.evidence) + 1}"
        self.evidence.append({"id": eid, "kind": kind, "city": city, "call": call, "data": data})
        return eid

    def run_tool(self, name, a):
        city = a.get("city")
        if name == "competitors":
            kws = a.get("keywords") or self.keywords
            data = self.q.competitors(city, kws)
            return self._add("competitors", city, data, f"/search geo={city} keywords={len(kws)} type=place")
        if name == "audience_affinity":
            raw = self.q.insights(a["domain"], tags=self.tags, location=city, take=12)
            data = [e for e in raw if not is_noise(e, city)][:8]
            return self._add(f"affinity:{a['domain']}", city, data,
                             f"/v2/insights filter.type={a['domain']} signal.tags=bubble_tea signal.location={city} (noise-filtered {len(raw)-len(data)})")
        if name == "audience_profile":
            ids = [x["id"] for k in ("affinity:artist", "affinity:movie") for e in [_ev(self.evidence, city, k)] if e for x in e["data"][:3]]
            profiles = self.q.demographics(ids[:6]) if ids else []
            data = [merge_profiles(profiles)] if profiles else []
            return self._add("profile", city, data, f"/v2/insights filter.type=demographics signal.entities={len(ids[:6])} top artists/films")
        if name == "hotspots":
            comp = _ev(self.evidence, city, "competitors")
            ids = [v["id"] for v in (comp["data"] if comp else [])[:5]]
            data = self.q.heatmap(ids, city) if ids else []
            return self._add("hotspots", city, data, f"/v2/insights filter.type=heatmap signal.entities={len(ids)} competitors location={city}")
        if name == "search_entity":
            data = self.q.search(a["query"], types=[a["type"]] if a.get("type") else None, take=5)
            return self._add("search", None, data, f"/search q='{a['query']}'")
        raise ValueError(name)

    def run(self, brief, on_event=None):
        emit = on_event or (lambda ev: None)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(brief, ensure_ascii=False)}]
        final = None
        for _ in range(self.max_steps):
            out = self.llm.chat(msgs, TOOLS)
            if not out["tool_calls"]:
                final = out["content"]; break
            msgs.append({"role": "assistant", "content": out["content"], "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in out["tool_calls"]]})
            for c in out["tool_calls"]:
                emit({"type": "tool_call", "name": c["name"], "args": c["arguments"]})
                try:
                    eid = self.run_tool(c["name"], c["arguments"])
                    ev = self.evidence[-1]
                    res = {"evidence_id": eid, "total_results": len(ev["data"]), "items_preview": ev["data"][:8]}
                    emit({"type": "evidence", "id": eid, "kind": ev["kind"], "city": ev["city"], "call": ev["call"], "count": len(ev["data"]),
                          "preview": [x.get("name") for x in ev["data"][:4] if isinstance(x, dict) and x.get("name")]})
                except Exception as e:
                    self.errors.append(f"{c['name']}({c['arguments']}): {e}")
                    emit({"type": "error", "name": c["name"], "message": str(e)[:200]})
                    res = {"error": str(e)}
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(res, ensure_ascii=False)[:6000]})
        report = synthesize(brief, self.evidence)
        report["errors"] = self.errors
        if final and final != "__SYNTHESIZE__":
            report["llm_narrative"] = self._grounded_narrative(brief, report, msgs) or final
        return report

    def _grounded_narrative(self, brief, report, msgs):
        """Final pass: make the LLM narrative agree with the deterministic, evidence-cited ranking."""
        facts = [{k: r.get(k) for k in ("city", "score", "parts", "venues", "avg_rating", "chinese_incumbents",
                                       "comp_eid", "prof_eid", "hot_eid", "artists", "brands", "movies", "adjacent")}
                 for r in report["ranking"]]
        ask = ("Final step. Below is the deterministic, evidence-cited scoring (authoritative; venue counts are totals). "
               "Write the market-entry plan for the brand in Markdown, <=350 words: recommend cities in EXACTLY this score order, "
               "explain why with the numbers, give 1 store-location, 2 collab, 1 menu and 1 content idea for the top city, and 1 risk per city. "
               "Cite evidence ids like [E1] only from this data. Do not invent numbers.\n" + json.dumps(facts, ensure_ascii=False))
        try:
            out = self.llm.chat([msgs[0], msgs[1], {"role": "user", "content": ask}])
            return (out.get("content") or "").strip() or None
        except Exception as e:
            self.errors.append(f"narrative: {e}")
            return None
        return report


def _ev(evidence, city, kind):
    return next((e for e in reversed(evidence) if e["city"] == city and e["kind"] == kind), None)


WEIGHTS = {"demand": 0.3, "whitespace": 0.3, "quality_gap": 0.1, "audience_fit": 0.3}


def synthesize(brief, evidence):
    """Deterministic, evidence-cited scoring (transparent for judges)."""
    rows = []
    for city in brief["cities"]:
        comp, prof, hot = (_ev(evidence, city, k) for k in ("competitors", "profile", "hotspots"))
        venues = comp["data"] if comp else []
        ratings = [v["rating"] for v in venues if v.get("rating")]
        cn = [v["name"] for v in venues if is_chinese_brand(v["name"])]
        aff = {d: _ev(evidence, city, f"affinity:{d}") for d in ("artist", "brand", "movie", "tag")}
        def names(d, n=4):
            e = aff[d]
            return ([x["name"] for x in e["data"][:n]], e["id"]) if e and e["data"] else ([], None)
        n, avg = len(venues), (round(mean(ratings), 2) if ratings else None)
        profile = prof["data"][0] if prof and prof["data"] else None
        fit = fit_score(profile)
        parts = {"demand": min(n, 30) / 30,
                 "whitespace": 1 - (len(cn) / n if n else 0),
                 "quality_gap": max(0, min(1, (5 - avg) / 1.5)) if avg else 0.5,
                 "audience_fit": fit if fit is not None else 0.5}
        score = round(100 * sum(WEIGHTS[k] * v for k, v in parts.items()))
        rows.append({"city": city, "score": score, "parts": {k: round(v, 2) for k, v in parts.items()},
                     "venues": n, "avg_rating": avg, "chinese_incumbents": cn, "comp_eid": comp["id"] if comp else None,
                     "profile": profile, "prof_eid": prof["id"] if prof else None,
                     "hotspot": hot["data"][0] if hot and hot["data"] else None, "hotspots": hot["data"] if hot else [], "hot_eid": hot["id"] if hot else None,
                     "top_venues": [f"{v['name'][:40]} ({v.get('rating','-')})" for v in venues[:3]],
                     "artists": names("artist"), "brands": names("brand"), "movies": names("movie"), "adjacent": names("tag")})
    rows.sort(key=lambda r: -r["score"])
    return {"brief": brief, "weights": WEIGHTS, "ranking": rows, "evidence": evidence}


def _skew(p):
    if not p:
        return "n/a"
    age = max(p["age"], key=p["age"].get) if p.get("age") else "?"
    g = p.get("gender", {}); gs = "female" if g.get("female", 0) > g.get("male", 0) else "male"
    return f"peak age {age.replace('_', ' ')}, {gs}-leaning ({g.get(gs, 0):+.2f})"


def render(report):
    b, rows = report["brief"], report["ranking"]
    L = [f"# TasteBridge: {b['brand']} ({b['category']}) — city entry comparison", "",
         "Score = 30% demand + 30% whitespace (few Chinese incumbents) + 10% quality gap + 30% audience fit", "",
         "| City | Score | Venues | Avg rating | Chinese incumbents | Audience fit | Evidence |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['city']} | {r['score']} | {r['venues']} | {r['avg_rating']} | {len(r['chinese_incumbents'])} | {r['parts']['audience_fit']} | [{r['comp_eid']}][{r['prof_eid']}] |")
    for r in rows:
        h = r["hotspot"]
        L += ["", f"## {r['city']}  (score {r['score']}: {r['parts']})",
              f"- Competitive set [{r['comp_eid']}]: {', '.join(r['top_venues']) or 'n/a'}",
              f"- Chinese incumbents [{r['comp_eid']}]: {', '.join(x[:30] for x in r['chinese_incumbents'][:5]) or 'none found'}",
              f"- Audience profile [{r['prof_eid']}]: {_skew(r['profile'])}",
              f"- Store hotspot [{r['hot_eid']}]: " + (f"{h['lat']:.4f},{h['lon']:.4f} (affinity {h['affinity']:.2f})" if h else "n/a")]
        for label, key in (("Music/KOL collab candidates", "artists"), ("Brand collab candidates", "brands"),
                           ("Film/TV for positioning & content", "movies"), ("Adjacent formats (menu/store concept)", "adjacent")):
            ns, eid = r[key]
            L.append(f"- {label} [{eid}]: {', '.join(ns) or 'n/a'}")
    top = rows[0]
    L += ["", "## Entry plan (draft; the LLM planner writes the narrative version)",
          f"1. Enter **{top['city']}** first (score {top['score']}) [{top['comp_eid']}][{top['prof_eid']}].",
          f"2. Collabs: {', '.join(top['brands'][0][:2]) or 'n/a'} [{top['brands'][1]}]; music/KOL: {', '.join(top['artists'][0][:2]) or 'n/a'} [{top['artists'][1]}].",
          f"3. Store/menu concept: pair tea with {', '.join(top['adjacent'][0][:3]) or 'n/a'} [{top['adjacent'][1]}].",
          f"4. Content positioning: reference {', '.join(top['movies'][0][:2]) or 'n/a'} [{top['movies'][1]}].",
          ""]
    if report.get("llm_narrative"):
        L += ["## LLM entry plan (grounded in the scores above)", report["llm_narrative"], ""]
    L += ["## Evidence log"]
    for err in report.get("errors", []):
        L.append(f"- TOOL ERROR: {err[:160]}")
    for e in report["evidence"]:
        L.append(f"- [{e['id']}] {e['call']} -> {len(e['data'])} results")
    return "\n".join(L)
