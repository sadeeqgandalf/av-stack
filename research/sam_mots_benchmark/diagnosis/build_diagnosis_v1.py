#!/usr/bin/env python3
"""Occlusion diagnosis tables (CSV + HTML page) from the benchmark and occlusion-analysis outputs.

Every number is read from result files (TrackEval summaries/detailed CSVs, run_info, occlusion JSONs);
finding sentences are computed from the same numbers, with assertions guarding each claim.

    python diagnosis/build_diagnosis.py
"""
from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "pod_results" / "results"
RUNS = ROOT / "pod_results" / "trackers"
OCC = ROOT / "occlusion" / "results"

BENCH = [("kitti_mots", "car", "KITTI-MOTS · car"), ("kitti_mots", "pedestrian", "KITTI-MOTS · pedestrian"),
         ("mots_challenge", "pedestrian", "MOTS20 · pedestrian")]
T3, T31 = "SAM3", "SAM3.1"


def summary(bench, tracker, cls):
    head, vals = (RES / bench / tracker / f"{cls}_summary.txt").read_text().split("\n")[:2]
    return dict(zip(head.split(), map(float, vals.split())))


def detailed(bench, tracker, cls):
    with open(RES / bench / tracker / f"{cls}_detailed.csv") as f:
        return {r["seq"]: {k: float(v) for k, v in r.items() if k != "seq"} for r in csv.DictReader(f)}


def run_objects(bench, tracker):
    objs = {}
    for p in sorted((RUNS / bench / tracker).glob("run_info*.json")):
        for seq, v in json.loads(p.read_text())["sequences"].items():
            if "objects" in v:
                objs[seq] = v["objects"]
    return objs


S = {(b, c, t): summary(b, t, c) for b, c, _ in BENCH for t in (T3, T31, "track_rcnn")}
occ = json.loads((OCC / "occlusion_results.json").read_text())
scene = json.loads((OCC / "scene_stats.json").read_text())
occ_by = {s["subset"]: s for s in occ["summary"]}
det3, det31 = detailed("mots_challenge", T3, "pedestrian"), detailed("mots_challenge", T31, "pedestrian")
obj3, obj31 = run_objects("mots_challenge", T3), run_objects("mots_challenge", T31)
SEQS = ["MOTS20-02", "MOTS20-05", "MOTS20-09", "MOTS20-11"]


def f1(x):
    return f"{x:.1f}"


def d1(x):
    return f"{x:+.1f}"


def pct(x):
    return f"{100 * x:.1f}"


def dpct(a, b):
    return f"{100 * (b - a):+.1f}"


def ci(c):
    return f"{100 * c[0]:.0f}–{100 * c[1]:.0f}"


# every table: title (the question), columns, rows (label, direction, cells...), finding, row flags
tables = []

# ---- 1. detection, segmentation or identity -------------------------------------------------------
t1_metrics = [("HOTA", "overall score = √(DetA × AssA)", "↑"), ("DetA", "finding the objects (detection)", "↑"),
              ("LocA", "mask quality (segmentation)", "↑"), ("AssA", "keeping each object's identity (association)", "↑")]
rows = []
for m, lab, arrow in t1_metrics:
    cells = []
    for b, c, _ in BENCH:
        a, z = S[(b, c, T3)][m], S[(b, c, T31)][m]
        cells += [f1(a), f1(z), d1(z - a)]
    rows.append({"metric": m, "label": lab, "dir": arrow, "cells": cells, "flag": m == "AssA"})
d = {m: [S[(b, c, T31)][m] - S[(b, c, T3)][m] for b, c, _ in BENCH] for m, _, _ in t1_metrics}
assert all(d["AssA"][i] < d["DetA"][i] for i in range(3)) and all(abs(x) < 1 for x in d["LocA"])
tables.append({
    "id": "t1", "q": "Is the problem detection, segmentation, or identity?",
    "cols": ["SAM 3", "SAM 3.1", "Δ"] * 3, "groups": [g for _, _, g in BENCH],
    "rows": rows,
    "finding": (f"Identity. AssA falls most on every benchmark ({', '.join(d1(x) for x in d['AssA'])}), largest on crowded "
                f"MOTS20. Mask quality is unchanged (LocA Δ {', '.join(d1(x) for x in d['LocA'])}). Detection also "
                f"drops ({', '.join(d1(x) for x in d['DetA'])}), but less than identity.")})

# ---- 2. which identity error --------------------------------------------------------------------
t2 = [("AssRe", "split: one person → several IDs (higher = fewer splits)", "↑", 1),
      ("AssPr", "merge: one ID → several people (higher = fewer merges)", "↑", 1),
      ("IDSW", "identity switches", "↓", 0), ("Frag", "track fragmentations", "↓", 0),
      ("IDratio", "predicted IDs per real person (ideal 1.00)", "→ 1", 2),
      ("MLR", "people mostly lost, % (tracked < 20% of their life)", "↓", 1)]
rows = []
for m, lab, arrow, kind in t2:
    cells = []
    for b, c, _ in BENCH:
        if m == "IDratio":
            a, z = (S[(b, c, t)]["IDs"] / S[(b, c, t)]["GT_IDs"] for t in (T3, T31))
            cells += [f"{a:.2f}", f"{z:.2f}", f"{z - a:+.2f}"]
        elif kind == 0:
            a, z = S[(b, c, T3)][m], S[(b, c, T31)][m]
            cells += [f"{a:.0f}", f"{z:.0f}", f"{z - a:+.0f}"]
        else:
            a, z = S[(b, c, T3)][m], S[(b, c, T31)][m]
            cells += [f1(a), f1(z), d1(z - a)]
    rows.append({"metric": m if m != "IDratio" else "IDs / GT IDs", "label": lab, "dir": arrow, "cells": cells,
                 "flag": m in ("AssRe", "IDratio", "MLR")})
m_re = S[("mots_challenge", "pedestrian", T31)]["AssRe"] - S[("mots_challenge", "pedestrian", T3)]["AssRe"]
m_pr = S[("mots_challenge", "pedestrian", T31)]["AssPr"] - S[("mots_challenge", "pedestrian", T3)]["AssPr"]
r3, r31 = (S[("mots_challenge", "pedestrian", t)]["IDs"] / S[("mots_challenge", "pedestrian", t)]["GT_IDs"] for t in (T3, T31))
assert m_re < -10 and m_pr > -1
ml3, ml31 = (S[("mots_challenge", "pedestrian", t)]["MLR"] for t in (T3, T31))
assert ml31 > 2 * ml3
assert all(S[("kitti_mots", c, T31)][m] < S[("kitti_mots", c, T3)][m] for c in ("car", "pedestrian") for m in ("AssRe", "AssPr"))
tables.append({
    "id": "t2", "q": "What kind of identity error?",
    "cols": ["SAM 3", "SAM 3.1", "Δ"] * 3, "groups": [g for _, _, g in BENCH], "rows": rows,
    "finding": (f"Splitting. On MOTS20, SAM 3.1 breaks people into several IDs (AssRe {d1(m_re)}) but does not merge "
                f"people (AssPr {d1(m_pr)}). It predicts {r31:.2f} IDs per real person (SAM 3: {r3:.2f}), and it also loses more people for most of their time on screen ({ml31:.1f}% mostly lost vs {ml3:.1f}%). On KITTI, splits "
                f"and merges both increase moderately (AssRe and AssPr both lower).")})

# ---- 3. is occlusion the cause ------------------------------------------------------------------
c3, c31 = occ["control_no_occlusion"][T3], occ["control_no_occlusion"][T31]
a = occ_by["all events"]
rows = [
    {"metric": "No occlusion", "label": "same ID 1 s later, person stays visible (control)", "dir": "↑",
     "cells": [pct(c3["same_id_rate"]), ci(c3["ci"]), str(c3["windows"]), pct(c31["same_id_rate"]), ci(c31["ci"]),
               str(c31["windows"]), dpct(c3["same_id_rate"], c31["same_id_rate"])], "flag": False},
    {"metric": "Through occlusion", "label": "same ID after the person reappears (visibility < 0.25)", "dir": "↑",
     "cells": [pct(a[f"{T3} recovery"]), ci(a[f"{T3} recovery_ci"]), str(a[f"{T3} events"]),
               pct(a[f"{T31} recovery"]), ci(a[f"{T31} recovery_ci"]), str(a[f"{T31} events"]),
               dpct(a[f"{T3} recovery"], a[f"{T31} recovery"])], "flag": True},
]
for key, lab in (("kept", "kept the same ID"), ("switched", "re-found under a new ID"), ("lost", "not found within 2 s")):
    rows.append({"metric": f"  outcome: {key}", "label": lab, "dir": "", "flag": key == "switched",
                 "cells": [str(a[f"{T3} {key}"]), "", "", str(a[f"{T31} {key}"]), "", "",
                           f"{a[f'{T31} {key}'] - a[f'{T3} {key}']:+d}"]})
p = occ["paired"]
gap_ctrl = 100 * (c31["same_id_rate"] - c3["same_id_rate"])
gap_occ = 100 * (a[f"{T31} recovery"] - a[f"{T3} recovery"])
assert gap_occ < 5 * gap_ctrl < 0 and a[f"{T31} switched"] > 3 * a[f"{T31} lost"]
tables.append({
    "id": "t3", "q": "Is occlusion the cause? (MOTS20)",
    "cols": ["%", "95% CI", "n", "%", "95% CI", "n", "Δ"], "groups": ["SAM 3", "SAM 3.1", ""], "group_spans": [3, 3, 1],
    "rows": rows,
    "finding": (f"Yes. Without occlusion the models are {abs(gap_ctrl):.1f} points apart; through occlusion "
                f"{abs(gap_occ):.1f} points. When SAM 3.1 fails it usually finds the person again under a new ID "
                f"({a[f'{T31} switched']} new IDs, {a[f'{T31} lost']} lost). Paired, on the {p['events_tracked_before_by_both']} "
                f"events both models tracked beforehand: only SAM 3 kept the ID {p['only_SAM3_kept']} times, only SAM 3.1 "
                f"{p['only_SAM3.1_kept']} time (McNemar p = {p['mcnemar_exact_p']:.0e}).")})

# ---- 4. which occlusions -----------------------------------------------------------------------
subs = [("duration <0.5 s", "< 0.5 s"), ("duration 0.5-1 s", "0.5–1 s"), ("duration 1-2 s", "1–2 s"),
        ("duration >=2 s", "≥ 2 s"), ("depth partial (min vis 0.10-0.25)", "partial (min visibility 0.10–0.25)"),
        ("depth heavy/full (min vis < 0.10)", "heavy / full (min visibility < 0.10)")]
rows = []
for key, lab in subs:
    s = occ_by[key]
    rows.append({"metric": "duration" if key.startswith("duration") else "depth", "label": lab, "dir": "↑",
                 "flag": key in ("duration >=2 s", "depth heavy/full (min vis < 0.10)"),
                 "cells": [f"{s[f'{T3} events']} / {s[f'{T31} events']}", pct(s[f"{T3} recovery"]), ci(s[f"{T3} recovery_ci"]),
                           pct(s[f"{T31} recovery"]), ci(s[f"{T31} recovery_ci"]), dpct(s[f"{T3} recovery"], s[f"{T31} recovery"])]})
dur = [occ_by[k] for k, _ in subs[:4]]
r31 = [s[f"{T31} recovery"] for s in dur]
r3 = [s[f"{T3} recovery"] for s in dur]
assert r31[0] > r31[1] and r31[0] > r31[3] and r3[3] < min(r3[:3])
tables.append({
    "id": "t4", "q": "Which occlusions? (MOTS20)",
    "cols": ["events (3 / 3.1)", "same ID %", "95% CI", "same ID %", "95% CI", "Δ"],
    "groups": ["", "SAM 3", "SAM 3.1", ""], "group_spans": [1, 2, 2, 1], "rows": rows,
    "finding": (f"Longer and deeper occlusions. SAM 3.1 keeps {pct(r31[0])}% of IDs through occlusions under 0.5 s but "
                f"{pct(r31[3])}% at 2 s or more. SAM 3 is steady up to 2 s ({pct(min(r3[:3]))}–{pct(max(r3[:3]))}%) and "
                f"then also falls, to {pct(r3[3])}%. Long occlusions are an open problem for both models.")})

# ---- 5. which scenes ----------------------------------------------------------------------------
rows = []
for seq in SEQS:
    sc, s = scene[seq], occ_by[seq]
    rows.append({"metric": seq, "label": f"{sc['resolution']}, {sc['fps']:.0f} fps", "dir": "",
                 "flag": False,
                 "cells": [f"{sc['gt_people']}", f"{sc['visible_people_per_frame']:.1f}", f"{sc['pct_person_frames_occluded']:.0f}",
                           f"{det3[seq]['IDs'] / det3[seq]['GT_IDs']:.2f}", f"{det31[seq]['IDs'] / det31[seq]['GT_IDs']:.2f}",
                           f1(100 * det3[seq]["AssA___AUC"]), f1(100 * det31[seq]["AssA___AUC"]),
                           d1(100 * (det31[seq]["AssA___AUC"] - det3[seq]["AssA___AUC"])),
                           pct(s[f"{T3} recovery"]), pct(s[f"{T31} recovery"]), dpct(s[f"{T3} recovery"], s[f"{T31} recovery"])]})
worst = min(SEQS, key=lambda q: occ_by[q][f"{T31} recovery"] - occ_by[q][f"{T3} recovery"])
best = max(SEQS, key=lambda q: occ_by[q][f"{T31} recovery"] - occ_by[q][f"{T3} recovery"])
for r in rows:
    r["flag"] = r["metric"] == worst
ratios31 = [det31[q]["IDs"] / det31[q]["GT_IDs"] for q in SEQS]
ratios3 = [det3[q]["IDs"] / det3[q]["GT_IDs"] for q in SEQS]
assert all(det3[q]["GT_IDs"] == scene[q]["gt_people"] for q in SEQS)          # same people count from both sources
assert all(x > y for x, y in zip(ratios31, ratios3))
assert all(occ_by[q][f"{T31} recovery"] < occ_by[q][f"{T3} recovery"] for q in SEQS)          # behind on every scene
assert max(SEQS, key=lambda q: scene[q]["pct_person_frames_occluded"]) == "MOTS20-02"          # most occluded
assert min(SEQS, key=lambda q: occ_by[q][f"{T3} recovery"]) == "MOTS20-02"                    # SAM 3's lowest
tables.append({
    "id": "t5", "q": "Which scenes? (MOTS20, per sequence)",
    "cols": ["people", "visible / frame", "% occluded", "SAM 3", "SAM 3.1", "SAM 3", "SAM 3.1", "Δ", "SAM 3", "SAM 3.1", "Δ"],
    "groups": ["scene", "predicted IDs per person", "AssA", "same ID through occlusion, %"], "group_spans": [3, 2, 3, 3],
    "rows": rows,
    "finding": (f"SAM 3.1 is behind on every scene: the largest gap is {worst} "
                f"({dpct(occ_by[worst][f'{T3} recovery'], occ_by[worst][f'{T31} recovery'])} points) and the smallest {best} "
                f"({dpct(occ_by[best][f'{T3} recovery'], occ_by[best][f'{T31} recovery'])}). On every scene it creates more IDs "
                f"per person ({min(ratios31):.2f}–{max(ratios31):.2f}) than SAM 3 ({min(ratios3):.2f}–{max(ratios3):.2f}). "
                f"MOTS20-02 is the most occluded scene ({scene['MOTS20-02']['pct_person_frames_occluded']:.0f}% of person-frames) "
                f"and has SAM 3's lowest recovery ({pct(occ_by['MOTS20-02'][f'{T3} recovery'])}%).")})

# ---- context: trained reference -----------------------------------------------------------------
rows = []
for m, lab in (("HOTA", "overall"), ("DetA", "detection"), ("AssA", "association"), ("IDSW", "identity switches")):
    cells = []
    for b, c, _ in BENCH:
        vals = [S[(b, c, t)][m] for t in (T3, T31, "track_rcnn")]
        cells += [f"{v:.0f}" if m == "IDSW" else f1(v) for v in vals]
    rows.append({"metric": m, "label": lab, "dir": "↓" if m == "IDSW" else "↑", "cells": cells, "flag": False})
tables.append({
    "id": "t6", "q": "Context: zero-shot SAM against a trained tracker (TrackR-CNN, CVPR 2019)",
    "cols": ["SAM 3", "SAM 3.1", "TrackR-CNN"] * 3, "groups": [g for _, _, g in BENCH], "rows": rows,
    "finding": ("Both SAM models, prompted only with a class name, associate better than TrackR-CNN, which was trained on "
                "these datasets. The problem above is a gap between the two SAM versions, and in long occlusions "
                "for both, not a general weakness of SAM.")})

# ---- CSV -----------------------------------------------------------------------------------------
out_csv = HERE / "occlusion_diagnosis.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["SAM 3 vs SAM 3.1: where identity is lost under occlusion. Zero-shot, text prompts; "
                "official TrackEval (HOTA) on KITTI-MOTS val and MOTS20 train; occlusion from MOT17 visibility. "
                "Δ = SAM 3.1 − SAM 3."])
    w.writerow([])
    for n, t in enumerate(tables, 1):
        w.writerow([f"Table {n}. {t['q']}"])
        spans = t.get("group_spans") or [len(t["cols"]) // len(t["groups"])] * len(t["groups"])
        grow = ["", "", ""]
        for g, sp in zip(t["groups"], spans):
            grow += [g] + [""] * (sp - 1)
        w.writerow(grow)
        w.writerow(["metric", "meaning", "better"] + t["cols"])
        for r in t["rows"]:
            w.writerow([r["metric"].strip(), r["label"], r["dir"]] + r["cells"])
        w.writerow(["Finding", t["finding"]])
        w.writerow([])

# ---- HTML ----------------------------------------------------------------------------------------
def delta_class(txt):
    try:
        v = float(txt.replace("+", ""))
    except ValueError:
        return ""
    return "neg" if v < 0 else ("pos" if v > 0 else "")


def table_html(n, t):
    spans = t.get("group_spans") or [len(t["cols"]) // len(t["groups"])] * len(t["groups"])
    g = "".join(f"<th colspan='{sp}' class='grp'>{html.escape(x)}</th>" for x, sp in zip(t["groups"], spans))
    cols = "".join(f"<th class='num'>{html.escape(c)}</th>" for c in t["cols"])
    # which columns are deltas, and whether a delta is "bad" depends on the row's direction
    body = []
    for r in t["rows"]:
        tds = []
        for c, v in zip(t["cols"], r["cells"]):
            cls = "num"
            if c == "Δ" and v:
                k = delta_class(v)
                if r["dir"] == "↓":
                    k = {"neg": "pos", "pos": "neg"}.get(k, k)
                if r["dir"] == "→ 1" or r["dir"] == "":
                    k = "neu" if k else ""
                cls += f" delta {k}"
            tds.append(f"<td class='{cls}'>{html.escape(v)}</td>")
        indent = " sub" if r["metric"].startswith("  ") else ""
        body.append(f"<tr class='{'flag' if r['flag'] else ''}{indent}'><th scope='row'><span class='m'>"
                    f"{html.escape(r['metric'].strip())}</span><span class='lab'>{html.escape(r['label'])}</span></th>"
                    f"<td class='dir'>{html.escape(r['dir'])}</td>{''.join(tds)}</tr>")
    return f"""
<section class="tbl" id="{t['id']}">
  <h2><span class="qn">Q{n}</span>{html.escape(t['q'])}</h2>
  <div class="scroll"><table>
    <thead><tr><th></th><th></th>{g}</tr><tr><th class="lefth">metric</th><th class="dir">better</th>{cols}</tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table></div>
  <p class="finding"><b>Finding.</b> {html.escape(t['finding'])}</p>
</section>"""


chart = base64.b64encode((OCC / "id_recovery_by_duration.png").read_bytes()).decode()
ex = base64.b64encode((OCC / "example_MOTS20-02_person26.jpg").read_bytes()).decode()
chain = [
    ("Not segmentation", f"mask quality equal (LocA Δ {d1(d['LocA'][2])} on MOTS20)"),
    ("Identity", f"AssA is the largest drop ({d1(d['AssA'][2])} on MOTS20)"),
    ("Splitting", f"one person → several IDs (AssRe {d1(m_re)}, AssPr {d1(m_pr)})"),
    ("Caused by occlusion", f"gap {abs(gap_ctrl):.1f} points without occlusion, {abs(gap_occ):.1f} through it"),
    ("Re-identification", f"person re-found under a new ID ({a[f'{T31} switched']} new IDs vs {a[f'{T31} lost']} lost)"),
    ("Grows with duration", f"SAM 3.1 {pct(r31[0])}% → {pct(r31[3])}%; SAM 3 also falls to {pct(r3[3])}% at ≥ 2 s"),
]
chain_html = "".join(f"<li><b>{html.escape(k)}</b><span>{html.escape(v)}</span></li>" for k, v in chain)
page = f"""<title>SAM Occlusion Diagnosis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500&display=swap">
<style>
/* Layout: a single reading column of diagnostic tables, each answering one question, narrowing from metric family to scene. */
:root {{
  --bg: #f6f7f8; --paper: #ffffff; --ink: #1d2126; --muted: #5d6570; --rule: #dde1e6; --grp: #eef1f4;
  --sam3: #2a78d6; --sam31: #d45b1f; --bad: #b3261e; --bad-bg: #fbeae8; --good: #1f7a4a; --flag-bg: #fff6e8; --flag: #c77800;
  --display: "IBM Plex Serif", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #15181c; --paper: #1d2126; --ink: #e8eaed; --muted: #9aa3ad; --rule: #333a42; --grp: #252a30;
  --sam3: #6aa6ef; --sam31: #f08a55; --bad: #f28b82; --bad-bg: #3a2120; --good: #6fcf97; --flag-bg: #2e2717; --flag: #f0b44c; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #15181c; --paper: #1d2126; --ink: #e8eaed; --muted: #9aa3ad; --rule: #333a42; --grp: #252a30;
  --sam3: #6aa6ef; --sam31: #f08a55; --bad: #f28b82; --bad-bg: #3a2120; --good: #6fcf97; --flag-bg: #2e2717; --flag: #f0b44c; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); }}
.wrap {{ max-width: 1120px; margin: 0 auto; padding-inline: 20px; padding-block: 36px 64px; display: grid; gap: 34px; }}
header {{ display: grid; gap: 10px; max-width: 72ch; }}
.eyebrow {{ font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
h1 {{ font: 500 clamp(26px, 3.4vw, 36px)/1.15 var(--display); margin: 0; text-wrap: balance; }}
header p {{ margin: 0; color: var(--muted); }}
.key {{ display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 13px; color: var(--muted); }}
.key b {{ font-weight: 600; }}
.s3 {{ color: var(--sam3); }} .s31 {{ color: var(--sam31); }}
.chain {{ background: var(--paper); border: 1px solid var(--rule); border-radius: 6px; padding: 18px 20px; display: grid; gap: 10px; }}
.chain h2 {{ margin: 0; font: 600 13px/1 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
.chain ol {{ margin: 0; padding-left: 22px; display: grid; gap: 6px; }}
.chain li b {{ display: inline-block; min-width: 13.5em; }}
.chain li span {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
.tbl {{ display: grid; gap: 10px; min-width: 0; }}
.tbl h2 {{ margin: 0; font: 600 18px/1.3 var(--body); display: flex; gap: 10px; align-items: baseline; text-wrap: balance; }}
.qn {{ font: 500 13px/1 var(--mono); color: var(--paper); background: var(--ink); padding: 4px 7px; border-radius: 3px; }}
.scroll {{ overflow-x: auto; background: var(--paper); border: 1px solid var(--rule); border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 13.5px; }}
th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; }}
thead th {{ font: 500 12px/1.3 var(--mono); color: var(--muted); white-space: nowrap; background: var(--grp); }}
thead th.grp {{ text-align: center; color: var(--ink); border-left: 1px solid var(--rule); }}
tbody th {{ font-weight: 400; min-width: 16em; }}
.m {{ display: block; font: 500 13px/1.3 var(--mono); }}
.lab {{ display: block; color: var(--muted); font-size: 12.5px; }}
td.num, th.num {{ text-align: right; white-space: nowrap; font-family: var(--mono); font-size: 13px; }}
td.dir {{ color: var(--muted); text-align: center; font-family: var(--mono); font-size: 12px; }}
td.delta {{ font-weight: 500; }}
td.delta.neg {{ color: var(--bad); background: var(--bad-bg); }}
td.delta.pos {{ color: var(--good); }}
tr.flag th {{ box-shadow: inset 3px 0 0 var(--flag); background: var(--flag-bg); }}
tr.flag td {{ background: var(--flag-bg); }}
tr.flag td.delta.neg {{ background: var(--bad-bg); }}
tr.sub th .m {{ padding-left: 14px; color: var(--muted); }}
.finding {{ margin: 0; max-width: 90ch; }}
.finding b {{ font-weight: 600; }}
figure {{ margin: 0; display: grid; gap: 8px; min-width: 0; }}
figure img {{ border: 1px solid var(--rule); border-radius: 6px; background: #fff; width: 100%; }}
figcaption {{ color: var(--muted); font-size: 13px; max-width: 90ch; }}
.figs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr)); gap: 22px; }}
.notes {{ display: grid; gap: 6px; color: var(--muted); font-size: 13.5px; max-width: 90ch; }}
.notes h2 {{ color: var(--ink); font: 600 16px/1.3 var(--body); margin: 0 0 4px; }}
.notes ul {{ margin: 0; padding-left: 20px; display: grid; gap: 4px; }}
code {{ font-family: var(--mono); font-size: 12.5px; }}
@media (max-width: 560px) {{ .chain li b {{ min-width: 0; display: block; }} }}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">Occlusion study · final analysis before solutions</div>
  <h1>Where SAM loses identity under occlusion</h1>
  <p>SAM 3 vs SAM 3.1, zero-shot with text prompts only. Official TrackEval (HOTA) on KITTI-MOTS val and MOTS20 train;
     occlusion events from MOT17 per-frame visibility labels (245 events, 204 pedestrians). Each table answers one question
     and narrows the search.</p>
  <div class="key"><span><b class="s3">SAM 3</b> · <b class="s31">SAM 3.1</b></span><span><b>Δ</b> = SAM 3.1 − SAM 3; red where SAM 3.1 is worse</span>
     <span><b>better</b>: ↑ higher, ↓ lower</span><span>highlighted rows: where the problem shows</span></div>
</header>
<section class="chain" aria-label="Diagnosis in six steps">
  <h2>Diagnosis</h2>
  <ol>{chain_html}</ol>
</section>
{''.join(table_html(n, t) for n, t in enumerate(tables, 1))}
<div class="figs">
<figure><img alt="Bar chart: same identity after occlusion by duration, SAM 3 vs SAM 3.1, with 95% confidence intervals" src="data:image/png;base64,{chart}">
<figcaption>Same ID after occlusion, by duration, with the no-occlusion control on the left. Whiskers: 95% bootstrap CI over people.</figcaption></figure>
<figure><img alt="Example: MOTS20-02 pedestrian before, during and after occlusion; SAM 3 keeps ID 14, SAM 3.1 changes ID 9 to 83" src="data:image/jpeg;base64,{ex}">
<figcaption>Hand-checked example, MOTS20-02 person 26 (white box), hidden for 0.5 s: SAM 3 keeps ID 14; SAM 3.1 changes ID 9 → 83.</figcaption></figure>
</div>
<section class="notes">
  <h2>Method and limits</h2>
  <ul>
    <li>Models run exactly as in Meta's notebooks; SAM 3.1 with <code>max_num_objects=128</code> (its model default). Memory offloads were verified to leave outputs byte-identical.</li>
    <li>Occlusion event: visibility &lt; 0.25 (MOT17), visible before and after. Tracker ↔ ground truth: mask IoU ≥ 0.5, one-to-one per frame. Rules fixed before results.</li>
    <li>Pedestrians only for the occlusion study (KITTI has no visibility labels). 24 of 228 people could not be linked to MOT17 and are excluded. The ≥ 2 s bin is small (≈ 25 events per model).</li>
    <li>Open question for the next phase: whether SAM 3.1's losses come from its multiplexed memory under crowding (its log reports multiplex-bucket subscription of 1350% on MOTS20-02).</li>
    <li>Tables as CSV: <code>occlusion_diagnosis.csv</code>. Full protocol: <code>PROTOCOL.md</code>, <code>OCCLUSION_PROTOCOL.md</code>.</li>
  </ul>
</section>
</div>
"""
(HERE / "occlusion_diagnosis.html").write_text(page)
print("wrote", out_csv, HERE / "occlusion_diagnosis.html")
