import argparse, json, sys
from pathlib import Path
from .qloo import Qloo
from .llm import from_env
from .agent import Agent, render

ap = argparse.ArgumentParser(prog="tastebridge")
ap.add_argument("--brand", default="Lumen Leaf (fictional Shenzhen tea brand)")
ap.add_argument("--category", default="bubble tea")
ap.add_argument("--cities", default="Los Angeles,London,Singapore")
ap.add_argument("--json", help="write full report JSON here")
ap.add_argument("--cache", default=".cache/qloo")
a = ap.parse_args()
cities = [c.strip() for c in a.cities.split(",")]
brief = {"brand": a.brand, "category": a.category, "cities": cities}
agent = Agent(Qloo(cache_dir=a.cache), from_env(cities, a.category))
rep = agent.run(brief)
print(render(rep))
if a.json:
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
