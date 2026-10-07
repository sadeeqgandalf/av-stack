"""Run a scenario and write a self-contained 3D replay viewer (HTML).

    PYTHONPATH=. .venv/bin/python scripts/record.py highway
    PYTHONPATH=. .venv/bin/python scripts/record.py --all          # every scenario + an index page
    open viewer/highway.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from avstack.config import StackConfig
from avstack.runner import run
from avstack.sim.scenarios import SCENARIOS
from avstack.viz.recorder import Recorder, export_html


def record(name: str, seed: int, out_dir: Path) -> dict:
    scenario, cfg = SCENARIOS[name](), StackConfig()
    rec = Recorder(scenario, cfg)
    result = run(scenario, cfg, seed=seed, on_step=rec.on_step)
    data = rec.to_dict(result)
    path = export_html(data, out_dir / f"{name}.html")
    size = path.stat().st_size / 1e6
    print(f"{'PASS' if result.success else 'FAIL'} {name:18s} {len(rec.frames)} frames -> {path} ({size:.1f} MB)")
    return {"name": name, "description": scenario.description, "success": result.success,
            "distance": round(result.distance), "time": round(result.sim_time, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="viewer")
    a = ap.parse_args()
    names = list(SCENARIOS) if a.all or not a.scenarios else a.scenarios
    out = Path(a.out)
    rows = [record(n, a.seed, out) for n in names]
    if len(rows) > 1:
        items = "".join(
            f'<a class="card" href="{r["name"]}.html"><b>{html.escape(r["name"])}</b>'
            f'<span class="{"ok" if r["success"] else "bad"}">{"PASS" if r["success"] else "FAIL"}</span>'
            f'<p>{html.escape(r["description"])}</p><small>{r["distance"]} m · {r["time"]} s</small></a>'
            for r in rows)
        (out / "index.html").write_text(INDEX.replace("__ITEMS__", items).replace("__JSON__", json.dumps(rows)))
        print(f"index -> {out / 'index.html'}")


INDEX = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AV Stack Replays</title><style>
:root{--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--line:rgba(255,255,255,.1);--good:#0ca30c;--bad:#d03b3b}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;padding:32px 16px}
main{max-width:980px;margin:auto}h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--ink2);margin:0 0 24px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
.card{display:block;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;color:inherit;text-decoration:none}
.card:hover{border-color:#3987e5}.card b{font-size:15px}.card span{float:right;font-size:12px;font-weight:600}
.ok{color:var(--good)}.bad{color:var(--bad)}.card p{color:var(--ink2);margin:6px 0}.card small{color:var(--muted)}
</style></head><body><main><h1>AV Stack · scenario replays</h1>
<p class="sub">Closed-loop runs of the full stack. Open one to scrub through perception, prediction, planning and control.</p>
<div class="grid">__ITEMS__</div></main></body></html>"""

if __name__ == "__main__":
    main()
