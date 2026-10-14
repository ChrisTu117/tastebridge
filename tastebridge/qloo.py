"""Thin Qloo hackathon API tool layer (stdlib only). Never logs the API key."""
import json, os, re, urllib.parse, urllib.request
from pathlib import Path

BASE = os.environ.get("QLOO_BASE_URL", "https://hackathon.api.qloo.com")
# Valid entity types for /search. Anything else (e.g. urn:entity:director)
# silently returns [] -- that was the cause of the earlier empty search.
ENTITY_TYPES = {"artist", "book", "brand", "destination", "movie", "person",
                "place", "podcast", "tv_show", "videogame", "actor", "author", "director", "locality"}
SEARCH_TYPES = {"artist", "book", "brand", "destination", "movie", "person",
                "place", "podcast", "tv_show", "videogame", "locality"}
ALIASES = {"director": "person", "actor": "person", "author": "person", "film": "movie", "tv": "tv_show"}
CJK = re.compile(r"[\u4e00-\u9fff]")
CHINESE_TEA_BRANDS = ["heytea", "chagee", "mixue", "naixue", "nayuki", "chabaidao", "gong cha",
                      "coco", "yi fang", "tiger sugar", "machi machi", "chicha san chen", "molly tea", "lelecha", "xing fu tang"]


def load_key():
    k = os.environ.get("QLOO_API_KEY")
    if not k:
        p = Path.home() / ".secrets" / "qloo.env"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip().startswith("QLOO_API_KEY="):
                    k = line.split("=", 1)[1].strip().strip('"\'')
    if not k:
        raise RuntimeError("QLOO_API_KEY not set")
    return k


def norm_type(t):
    t = t.replace("urn:entity:", "")
    t = ALIASES.get(t, t)
    if t not in SEARCH_TYPES:
        raise ValueError(f"unsupported entity type {t!r}")
    return "urn:entity:" + t


class QlooError(RuntimeError):
    pass


class Qloo:
    def __init__(self, key=None, base=BASE, transport=None, cache_dir=None):
        self._key = key
        self.base = base.rstrip("/")
        self.transport = transport or self._http
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.backoff = 1.0
        self.calls = []  # (path, params) audit trail for evidence

    def __repr__(self):
        return f"Qloo(base={self.base!r})"  # never expose key

    def _http(self, url):
        req = urllib.request.Request(url, headers={"X-Api-Key": self._key or load_key(), "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, {"error": e.read()[:300].decode("utf-8", "replace")}

    def get(self, path, params):
        params = {k: v for k, v in params.items() if v is not None}
        qs = urllib.parse.urlencode(params, doseq=True)
        url = f"{self.base}{path}?{qs}"
        cache = None
        if self.cache_dir:
            import hashlib
            cache = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")
            if cache.exists():
                self.calls.append((path, params))
                return json.loads(cache.read_text())
        import time
        for attempt in range(4):  # retry 429/5xx/network errors with backoff
            try:
                status, data = self.transport(url)
            except Exception as e:  # timeouts, connection resets
                status, data = 599, {"error": type(e).__name__}
            if status not in (429, 500, 502, 503, 504, 599):
                break
            time.sleep(self.backoff * (2 ** attempt))
        if status != 200:
            raise QlooError(f"{path} -> HTTP {status}: {str(data)[:200]}")
        self.calls.append((path, params))
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
        return data

    # ---- tools ----
    def search(self, query, types=None, take=10):
        """Entity search. Unknown types are mapped (director->person) or rejected
        instead of silently returning nothing; falls back to untyped search."""
        t = ",".join(norm_type(x) for x in types) if types else None
        res = self.get("/search", {"query": query, "types": t, "take": take}).get("results", [])
        if not res and t:
            res = self.get("/search", {"query": query, "take": take}).get("results", [])
        return [simplify(e) for e in res]

    def find_tags(self, query, take=10):
        d = self.get("/v2/tags", {"filter.query": query, "take": take})
        return [{"id": t["id"], "name": t["name"], "type": t.get("type")} for t in d.get("results", {}).get("tags", [])]

    def insights(self, filter_type, tags=None, entities=None, location=None, take=10, extra=None):
        """Cross-domain affinity: what does an audience defined by tags/entities
        in `location` also like, within `filter_type`."""
        ft = filter_type if filter_type.startswith("urn:") else ("urn:tag" if filter_type == "tag" else norm_type(filter_type))
        p = {"filter.type": ft, "signal.interests.tags": ",".join(tags) if tags else None,
             "signal.interests.entities": ",".join(entities) if entities else None,
             "signal.location.query": location, "take": take}
        p.update(extra or {})
        r = self.get("/v2/insights", p).get("results", {})
        items = r.get("entities") or r.get("tags") or r.get("demographics") or []
        return [simplify(e) for e in items]

    def competitors(self, city, keywords, take=25, radius_km=25):
        """Local competitor scan. /search ignores the city in free text (it returns
        global matches), so we geo-filter with filter.location=lat,lon +
        filter.radius, run several keywords, dedupe by id and keep in-city hits."""
        ll = CITY_COORDS.get(city.lower())
        seen, out = set(), []
        for kw in keywords:
            p = {"query": kw, "types": "urn:entity:place", "take": take}
            if ll:
                p.update({"filter.location": f"{ll[0]},{ll[1]}", "filter.radius": radius_km})
            else:
                p["query"] = f"{kw} {city}"
            for e in map(simplify, self.get("/search", p).get("results", [])):
                if e.get("id") in seen or not in_city(e, city):
                    continue
                seen.add(e.get("id")); e["matched"] = kw; out.append(e)
        return out

    def demographics(self, entity_ids):
        """Age/gender skew (-1..1) of an audience. Works for media/brand/artist
        entities; returns [] for place or tag signals (verified on hackathon API)."""
        if not entity_ids:
            return []
        r = self.get("/v2/insights", {"filter.type": "urn:demographics",
                                      "signal.interests.entities": ",".join(entity_ids)}).get("results", {})
        return [d.get("query", {}) for d in r.get("demographics", [])]

    def heatmap(self, entity_ids, location, take=5):
        """Geo hotspots of affinity for entities within a city (needs entity signal)."""
        r = self.get("/v2/insights", {"filter.type": "urn:heatmap", "signal.interests.entities": ",".join(entity_ids),
                                      "filter.location.query": location, "take": take}).get("results", {})
        return [{"lat": h["location"]["latitude"], "lon": h["location"]["longitude"],
                 "affinity": h.get("query", {}).get("affinity")} for h in r.get("heatmap", [])[:take]]

    def compare(self, a_ids, b_ids, take=15):
        """Analysis Compare: taste themes (tags) the two audiences share, with a
        similarity score. Needs entity signals on both sides (verified)."""
        if not a_ids or not b_ids:
            return []
        r = self.get("/v2/insights/compare", {"a.signal.interests.entities": ",".join(a_ids),
                                              "b.signal.interests.entities": ",".join(b_ids), "take": take}).get("results", {})
        seen, out = set(), []
        for t in r.get("tags", []):
            n = t.get("name")
            if n and n.lower() not in seen and n.lower() not in GENERIC_TAGS:
                seen.add(n.lower()); out.append({"name": n, "score": round((t.get("query") or {}).get("score", 0), 3)})
        return out

    def trending(self, entity_id, filter_type, start, end):
        """Weekly trending series for one entity (population percentile and its
        week-over-week change). The hackathon data needs a window of a few months."""
        ft = filter_type if filter_type.startswith("urn:") else "urn:entity:" + filter_type
        r = self.get("/v2/trending", {"filter.type": ft, "signal.interests.entities": entity_id,
                                      "filter.start_date": start, "filter.end_date": end}).get("results", {})
        pts = sorted(r.get("trending", []), key=lambda x: x.get("date", ""))
        return [{"date": x["date"], "percentile": x.get("population_percentile"),
                 "delta": x.get("population_percent_delta")} for x in pts]


CITY_COORDS = {"singapore": (1.3521, 103.8198), "london": (51.5072, -0.1276), "los angeles": (34.0522, -118.2437),
               "new york": (40.7128, -74.006), "sydney": (-33.8688, 151.2093), "toronto": (43.6532, -79.3832),
               "kuala lumpur": (3.139, 101.6869), "tokyo": (35.6762, 139.6503), "paris": (48.8566, 2.3522)}
# LA metro sub-cities Qloo geocodes separately
CITY_ALIASES = {"los angeles": ["los angeles", "santa monica", "pasadena", "alhambra", "arcadia", "san gabriel",
                                "monterey park", "rowland heights", "irvine", "glendale", "burbank", "torrance", "culver city"]}


def in_city(e, city):
    c = city.lower()
    names = CITY_ALIASES.get(c, [c])
    hay = ((e.get("city") or "") + " " + (e.get("address") or "")).lower()
    return any(n in hay for n in names)


NOISE_NAMES = re.compile(r"[“\"].+[”\"]|\bmusk\b", re.I)  # nicknamed / meme entities
GENERIC_TAGS = {"place", "mastercard", "visa", "credit cards", "debit cards", "onsite services", "tourist attraction",
                "groups of friends", "moderate", "casual", "commercial", "nfc mobile payments"}


def is_noise(e, city=None):
    n = (e.get("name") or "").strip()
    if not n or NOISE_NAMES.search(n) or n.lower() in GENERIC_TAGS:
        return True
    if city and e.get("type", "").endswith("brand") and n.lower().startswith(city.lower()):
        return True  # civic brands named after the city (LA Times) are not collab targets for a tea shop
    return False


def simplify(e):
    p = e.get("properties") or {}
    geo = p.get("geocode") or {}
    return {k: v for k, v in {
        "id": e.get("entity_id") or e.get("tag_id") or e.get("id"),
        "name": e.get("name"),
        "type": e.get("subtype") or (e.get("types") or [e.get("type")])[0],
        "rating": p.get("business_rating"),
        "city": geo.get("city"),
        "address": p.get("address"),
        "popularity": e.get("popularity"),
        "affinity": (e.get("query") or {}).get("affinity"),
        "tags": [t.get("name") for t in (e.get("tags") or [])][:8],
    }.items() if v not in (None, [], "")}


def is_chinese_brand(name):
    n = name.lower()
    return bool(CJK.search(name)) or any(re.search(r'\b' + re.escape(b) + r'\b', n) for b in CHINESE_TEA_BRANDS)
