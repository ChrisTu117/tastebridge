import json, threading, unittest, urllib.request
from http.server import ThreadingHTTPServer
from tastebridge import server
from tastebridge.qloo import Qloo
from tastebridge.agent import Agent
from tastebridge.llm import MockPlanner
from tests.test_qloo import Fake, PLACE

SECRET = "sk-test-SECRET-123"


def fake_agent(cities, category):
    def routes(p, q):
        if p == "/search": return 200, {"results": [PLACE]}
        if q.get("filter.type") == "urn:heatmap":
            return 200, {"results": {"heatmap": [{"location": {"latitude": 51.5, "longitude": -0.1}, "query": {"affinity": 1}}]}}
        return 200, {"results": {"entities": [{"name": "Thing", "entity_id": "1"}]}}
    return Agent(Qloo(key=SECRET, transport=Fake(routes)), MockPlanner(cities, category))


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server.H.agent_factory = staticmethod(fake_agent)
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), server.H)
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls): cls.srv.shutdown()

    def post(self, body):
        req = urllib.request.Request(self.base + "/api/run", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req)

    def test_index_and_health(self):
        self.assertIn(b"TasteBridge", urllib.request.urlopen(self.base + "/").read())
        self.assertTrue(json.loads(urllib.request.urlopen(self.base + "/api/health").read())["ok"])

    def test_run_streams_trace_and_report_without_key(self):
        raw = self.post({"brand": "B", "cities": ["London"]}).read().decode()
        evs = [json.loads(l) for l in raw.splitlines()]
        types = [e["type"] for e in evs]
        self.assertEqual(types[0], "start"); self.assertIn("tool_call", types); self.assertEqual(types[-1], "report")
        self.assertEqual(evs[-1]["report"]["ranking"][0]["coords"], [51.5072, -0.1276])
        self.assertTrue(evs[-1]["report"]["ranking"][0]["hotspots"])
        self.assertNotIn(SECRET, raw)

    def test_validation(self):
        with self.assertRaises(urllib.error.HTTPError) as c: self.post({"cities": []})
        self.assertEqual(c.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError): self.post({"cities": list("abcde")})

    def test_static_has_no_key_reference(self):
        html = (server.STATIC / "index.html").read_text()
        self.assertNotIn("QLOO_API_KEY", html); self.assertNotIn("X-Api-Key", html)


if __name__ == "__main__": unittest.main()
