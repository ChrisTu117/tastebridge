import json, unittest, urllib.parse
from tastebridge.qloo import Qloo, norm_type, is_chinese_brand, simplify
from tastebridge.agent import Agent, synthesize, render
from tastebridge.llm import MockPlanner

PLACE = {"name": "Cuppacha Bubble Tea", "entity_id": "P1", "types": ["urn:entity:place"],
         "properties": {"business_rating": 4.3, "geocode": {"city": "London"}}}


class Fake:
    def __init__(self, routes): self.routes, self.urls = routes, []
    def __call__(self, url):
        self.urls.append(url)
        u = urllib.parse.urlparse(url); q = dict(urllib.parse.parse_qsl(u.query))
        return self.routes(u.path, q)


class TestTools(unittest.TestCase):
    def test_norm_type_maps_director(self):
        self.assertEqual(norm_type("director"), "urn:entity:person")
        with self.assertRaises(ValueError): norm_type("spaceship")

    def test_search_falls_back_untyped(self):
        f = Fake(lambda p, q: (200, {"results": [] if "types" in q else [PLACE]}))
        r = Qloo(key="k", transport=f).search("x", types=["artist"])
        self.assertEqual(r[0]["name"], "Cuppacha Bubble Tea"); self.assertEqual(len(f.urls), 2)

    def test_insights_params(self):
        f = Fake(lambda p, q: (200, {"results": {"entities": [{"name": "A", "entity_id": "1", "query": {"affinity": 0.9}}]}}))
        r = Qloo(key="k", transport=f).insights("artist", tags=["t1"], location="London")
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(f.urls[0]).query))
        self.assertEqual(q["filter.type"], "urn:entity:artist"); self.assertEqual(q["signal.location.query"], "London")
        self.assertEqual(r[0]["affinity"], 0.9)

    def test_tag_insights(self):
        f = Fake(lambda p, q: (200, {"results": {"tags": [{"tag_id": "urn:tag:x", "name": "Cafe"}]}}))
        self.assertEqual(Qloo(key="k", transport=f).insights("tag", tags=["t"])[0]["name"], "Cafe")
        self.assertIn("filter.type=urn%3Atag", f.urls[0])

    def test_competitors_city_filter(self):
        other = dict(PLACE, properties={"geocode": {"city": "Atlanta"}})
        f = Fake(lambda p, q: (200, {"results": [PLACE, other]}))
        self.assertEqual(len(Qloo(key="k", transport=f).competitors("London", ["bubble tea"])), 1)

    def test_http_error_raises(self):
        from tastebridge.qloo import QlooError
        with self.assertRaises(QlooError): Qloo(key="k", transport=lambda u: (401, {})).search("x")

    def test_key_not_in_repr(self):
        self.assertNotIn("secret", repr(Qloo(key="secret")))

    def test_chinese_brand(self):
        self.assertTrue(is_chinese_brand("Golden Tea 凰茶")); self.assertTrue(is_chinese_brand("HEYTEA Soho"))
        self.assertFalse(is_chinese_brand("Biju"))

    def test_agent_end_to_end_mock(self):
        def routes(p, q):
            if p == "/search": return 200, {"results": [PLACE]}
            return 200, {"results": {"entities": [{"name": "Thing", "entity_id": "1"}]}}
        rep = Agent(Qloo(key="k", transport=Fake(routes)), MockPlanner(["London"], "bubble tea")).run(
            {"brand": "B", "category": "bubble tea", "cities": ["London"]})
        self.assertEqual(len(rep["evidence"]), 9)  # target + 7 per-city + momentum
        md = render(rep); self.assertIn("| London |", md); self.assertIn("[E1]", md)


class TestCompareTrending(unittest.TestCase):
    def test_compare_dedupes_and_requires_entities(self):
        def routes(p, q):
            self.assertEqual(p, "/v2/insights/compare"); self.assertIn("a.signal.interests.entities", q)
            t = {"name": "Vegetarian", "query": {"score": 0.78}}
            return 200, {"results": {"tags": [t, t, {"name": "Casual", "query": {"score": 0.5}}, {"name": "Cozy", "query": {"score": 0.6}}]}}
        q = Qloo(key="k", transport=Fake(routes))
        self.assertEqual(q.compare(["a"], ["b"]), [{"name": "Vegetarian", "score": 0.78}, {"name": "Cozy", "score": 0.6}])
        self.assertEqual(q.compare([], ["b"]), [])

    def test_trending_sorted_and_momentum(self):
        from tastebridge.agent import momentum_score
        def routes(p, q):
            self.assertEqual(p, "/v2/trending"); self.assertIn("filter.start_date", q)
            return 200, {"results": {"trending": [{"date": "2026-10-01", "population_percentile": 0.6},
                                                   {"date": "2026-04-01", "population_percentile": 0.5}]}}
        s = Qloo(key="k", transport=Fake(routes)).trending("x", "artist", "2026-04-01", "2026-10-01")
        self.assertEqual(s[0]["date"], "2026-04-01")
        self.assertAlmostEqual(momentum_score([s]), round(0.7 * 0.6 + 0.3 * 0.7, 3))
        self.assertIsNone(momentum_score([]))

    def test_fit_uses_qloo_target(self):
        prof = {"age": {"24_and_younger": 0.5, "55_and_older": -0.5}, "gender": {"female": 0.3, "male": -0.3}}
        self.assertGreater(fit_score(prof, prof), 0.99)
        flipped = {"age": {"24_and_younger": -0.5, "55_and_older": 0.5}, "gender": {"female": -0.3, "male": 0.3}}
        self.assertLess(fit_score(prof, flipped), 0.01)


if __name__ == "__main__": unittest.main()


from tastebridge.qloo import in_city, is_noise
from tastebridge.agent import fit_score, merge_profiles


class TestPhase2(unittest.TestCase):
    def test_competitors_geo_multi_keyword_dedupe(self):
        sg = {"name": "HEYTEA ION", "entity_id": "S1", "types": ["urn:entity:place"], "properties": {"geocode": {"city": "Singapore"}}}
        jb = {"name": "Tea JB", "entity_id": "J1", "types": ["urn:entity:place"], "properties": {"geocode": {"city": "Johor Bahru"}}}
        f = Fake(lambda p, q: (200, {"results": [sg, jb]}))
        r = Qloo(key="k", transport=f).competitors("Singapore", ["bubble tea", "HEYTEA"])
        self.assertEqual([e["id"] for e in r], ["S1"])
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(f.urls[0]).query))
        self.assertEqual(q["filter.location"], "1.3521,103.8198"); self.assertIn("filter.radius", q)
        self.assertEqual(len(f.urls), 2)

    def test_la_metro_alias(self):
        self.assertTrue(in_city({"city": "Rowland Heights"}, "Los Angeles"))
        self.assertFalse(in_city({"city": "Lubin"}, "Los Angeles"))

    def test_noise_filter(self):
        self.assertTrue(is_noise({"name": "Elon “EDM” Musk"}))
        self.assertTrue(is_noise({"name": "Place"}))
        self.assertTrue(is_noise({"name": "Los Angeles Times", "type": "urn:entity:brand"}, "Los Angeles"))
        self.assertFalse(is_noise({"name": "Selfridges", "type": "urn:entity:brand"}, "London"))

    def test_brand_word_boundary(self):
        self.assertFalse(is_chinese_brand("Cocoa Bean Cafe")); self.assertTrue(is_chinese_brand("CoCo Fresh Tea"))

    def test_demographics_and_heatmap(self):
        def routes(p, q):
            if q["filter.type"] == "urn:demographics":
                return 200, {"results": {"demographics": [{"query": {"age": {"24_and_younger": 0.4}, "gender": {"female": 0.2, "male": -0.2}}}]}}
            return 200, {"results": {"heatmap": [{"location": {"latitude": 1.3, "longitude": 103.8}, "query": {"affinity": 0.9}}]}}
        q = Qloo(key="k", transport=Fake(routes))
        self.assertEqual(q.demographics(["a"])[0]["age"]["24_and_younger"], 0.4)
        self.assertEqual(q.demographics([]), [])
        self.assertEqual(q.heatmap(["a"], "Singapore")[0]["affinity"], 0.9)

    def test_fit_score(self):
        young = {"age": {"24_and_younger": 0.5, "55_and_older": -0.5}, "gender": {"female": 0.3, "male": -0.3}}
        old = {"age": {"24_and_younger": -0.5, "55_and_older": 0.5}, "gender": {"female": -0.3, "male": 0.3}}
        self.assertGreater(fit_score(young), 0.5); self.assertLess(fit_score(old), 0.5); self.assertIsNone(fit_score(None))
        m = merge_profiles([young, old]); self.assertEqual(m["age"]["24_and_younger"], 0)

    def test_retry_on_429(self):
        seq = iter([(429, {}), (200, {"results": []})])
        q = Qloo(key="k", transport=lambda u: next(seq)); q.backoff = 0
        self.assertEqual(q.search("x"), [])
