#!/usr/bin/env python3
"""Occlusion diagnosis: one shareable page (HTML, print-safe) + CSV, generated from the result files.

Sources: TrackEval summaries / per-sequence CSVs (pod_results/results), occlusion analysis
(occlusion/results/occlusion_results.json, scene_stats.json). Finding sentences are computed from the
same numbers and every claim they make is asserted, so the text cannot drift from the data.

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
OCC = ROOT / "occlusion" / "results"

TRK = [("SAM3", "SAM 3"), ("SAM3.1", "SAM 3.1"), ("track_rcnn", "TrackR-CNN")]
BENCH = [("kitti_mots", "car", "KITTI-MOTS val", "car"), ("kitti_mots", "pedestrian", "KITTI-MOTS val", "pedestrian"),
         ("mots_challenge", "pedestrian", "MOTS20 train", "pedestrian")]
SEQS = ["MOTS20-02", "MOTS20-05", "MOTS20-09", "MOTS20-11"]


def summary(b, t, c):
    head, vals = (RES / b / t / f"{c}_summary.txt").read_text().split("\n")[:2]
    return dict(zip(head.split(), map(float, vals.split())))


def detailed(b, t, c):
    with open(RES / b / t / f"{c}_detailed.csv") as f:
        return {r["seq"]: {k: float(v) for k, v in r.items() if k != "seq"} for r in csv.DictReader(f)}


S = {(b, c, t): summary(b, t, c) for b, c, _, _ in BENCH for t, _ in TRK}
DT = {t: detailed("mots_challenge", t, "pedestrian") for t, _ in TRK}
occ = json.loads((OCC / "occlusion_results.json").read_text())
scene = json.loads((OCC / "scene_stats.json").read_text())
OB = {s["subset"]: s for s in occ["summary"]}
CT = occ["control_no_occlusion"]
PR = occ["paired"]
MOTS = ("mots_challenge", "pedestrian")


def v(b, c, t, m):
    x = S[(b, c, t)]
    return x["IDs"] / x["GT_IDs"] if m == "IDratio" else x[m]


def rec(key, t):
    return OB[key][f"{t} recovery"] * 100


# ------------------------------------------------------------------ findings (computed + guarded)
dA = [v(b, c, "SAM3.1", "AssA") - v(b, c, "SAM3", "AssA") for b, c, _, _ in BENCH]
dD = [v(b, c, "SAM3.1", "DetA") - v(b, c, "SAM3", "DetA") for b, c, _, _ in BENCH]
dL = [v(b, c, "SAM3.1", "LocA") - v(b, c, "SAM3", "LocA") for b, c, _, _ in BENCH]
assert all(x < y for x, y in zip(dA, dD)) and all(abs(x) < 1 for x in dL)
dRe = v(*MOTS, "SAM3.1", "AssRe") - v(*MOTS, "SAM3", "AssRe")
dPr = v(*MOTS, "SAM3.1", "AssPr") - v(*MOTS, "SAM3", "AssPr")
assert dRe < -10 and dPr > 0
ctrl = {t: CT[t]["same_id_rate"] * 100 for t, _ in TRK}
allr = {t: rec("all events", t) for t, _ in TRK}
gap_c, gap_o = ctrl["SAM3.1"] - ctrl["SAM3"], allr["SAM3.1"] - allr["SAM3"]
assert gap_o < 5 * gap_c < 0
A = OB["all events"]
assert A["SAM3.1 switched"] > 3 * A["SAM3.1 lost"] and A["track_rcnn switched"] > 3 * A["track_rcnn lost"]
durk = ["duration <0.5 s", "duration 0.5-1 s", "duration 1-2 s", "duration >=2 s"]
r3, r31, rtr = ([rec(k, t) for k in durk] for t in ("SAM3", "SAM3.1", "track_rcnn"))
assert r3[3] < min(r3[:3]) and r31[3] < r31[0] and all(rtr[i] < r31[i] < r3[i] for i in range(4))
assert all(allr[t] < ctrl[t] for t, _ in TRK)
assert all(min(range(4), key=lambda i: x[i]) == 3 for x in (r3, r31, rtr))      # every tracker lowest at >= 2 s
assert all(v(b, c, "SAM3", "AssA") > v(b, c, "track_rcnn", "AssA") and v(b, c, "SAM3.1", "AssA") > v(b, c, "track_rcnn", "AssA")
           for b, c, _, _ in BENCH)
worst = min(SEQS, key=lambda q: rec(q, "SAM3.1") - rec(q, "SAM3"))
assert all(rec(q, "SAM3.1") < rec(q, "SAM3") for q in SEQS)

FINDINGS = [
    ("Identity, not masks.",
     f"Association accuracy (AssA) is where SAM 3.1 falls furthest behind SAM 3 on every benchmark "
     f"(Δ {dA[0]:+.1f} / {dA[1]:+.1f} / {dA[2]:+.1f}); mask quality is unchanged (LocA Δ within ±{max(abs(x) for x in dL):.1f})."),
    ("People are split into several identities.",
     f"On MOTS20, association recall drops (AssRe Δ {dRe:+.1f}) while association precision does not (AssPr Δ {dPr:+.1f}): "
     f"SAM 3.1 outputs {v(*MOTS, 'SAM3.1', 'IDratio'):.2f} identities per person (SAM 3 {v(*MOTS, 'SAM3', 'IDratio'):.2f})."),
    ("The loss is caused by occlusion.",
     f"Without occlusion the ID is kept {ctrl['SAM3']:.1f}% / {ctrl['SAM3.1']:.1f}% / {ctrl['track_rcnn']:.1f}% of the time "
     f"(SAM 3 / SAM 3.1 / TrackR-CNN); through an occlusion {allr['SAM3']:.1f}% / {allr['SAM3.1']:.1f}% / {allr['track_rcnn']:.1f}%. "
     f"The SAM 3.1 − SAM 3 gap grows from {gap_c:+.1f} to {gap_o:+.1f} points (paired McNemar p = {PR['mcnemar_exact_p']:.1e})."),
    ("Failures are re-identifications.",
     f"When the ID is not kept, the person is usually found again under a new ID "
     f"(new ID / lost: SAM 3.1 {A['SAM3.1 switched']} / {A['SAM3.1 lost']}, TrackR-CNN {A['track_rcnn switched']} / {A['track_rcnn lost']})."),
    ("Longer occlusions are the open problem.",
     f"For occlusions of 2 s or more the ID is kept {r3[3]:.1f}% / {r31[3]:.1f}% / {rtr[3]:.1f}% of the time; "
     f"SAM 3 is the strongest tracker but also degrades there."),
]

# ------------------------------------------------------------------ table definitions
A_COLS = [("HOTA", "HOTA", 1, "↑"), ("DetA", "DetA", 1, "↑"), ("AssA", "AssA", 1, "↑"), ("LocA", "LocA", 1, "↑"),
          ("AssRe", "AssRe", 1, "↑"), ("AssPr", "AssPr", 1, "↑"), ("IDSW", "IDSW", 0, "↓"), ("Frag", "Frag", 0, "↓"),
          ("IDratio", "IDs / GT IDs", 2, "→1"), ("MLR", "ML %", 1, "↓")]
A_KEY = {"AssA", "AssRe"}
B_ROWS = [("control", "No occlusion (control)", "same ID 1 s later; person stays visible"),
          ("all events", "All occlusion events", "visibility < 0.25, then visible again"),
          ("duration <0.5 s", "Duration < 0.5 s", ""), ("duration 0.5-1 s", "0.5 – 1 s", ""),
          ("duration 1-2 s", "1 – 2 s", ""), ("duration >=2 s", "≥ 2 s", ""),
          ("depth partial (min vis 0.10-0.25)", "Depth: partial", "min visibility 0.10 – 0.25"),
          ("depth heavy/full (min vis < 0.10)", "Depth: heavy / full", "min visibility < 0.10")]


def fmt(x, dec):
    return f"{x:.{dec}f}"


def fdelta(x, dec):
    return f"{x:+.{dec}f}"


def bcell(key, t):
    if key == "control":
        c = CT[t]
        return c["same_id_rate"] * 100, [x * 100 for x in c["ci"]], c["windows"]
    s = OB[key]
    return s[f"{t} recovery"] * 100, [x * 100 for x in s[f"{t} recovery_ci"]], s[f"{t} events"]


# ------------------------------------------------------------------ CSV
out_csv = HERE / "occlusion_diagnosis.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Where SAM loses identity under occlusion: SAM 3 vs SAM 3.1 (zero-shot, text prompts) with TrackR-CNN as benchmark "
                "reference. Official TrackEval (HOTA/CLEAR/Identity). Δ = SAM 3.1 − SAM 3."])
    w.writerow([])
    w.writerow(["Table A. Benchmark results (TrackEval, combined over sequences)"])
    w.writerow(["benchmark", "class", "tracker"] + [lab for _, lab, _, _ in A_COLS])
    for b, c, bl, cl in BENCH:
        for t, tn in TRK:
            w.writerow([bl, cl, tn] + [fmt(v(b, c, t, m), d) for m, _, d, _ in A_COLS])
        w.writerow([bl, cl, "Δ (SAM 3.1 − SAM 3)"] + [fdelta(v(b, c, "SAM3.1", m) - v(b, c, "SAM3", m), d) for m, _, d, _ in A_COLS])
    w.writerow([])
    w.writerow(["Table B. Identity kept through occlusion, MOTS20 train (ID recovery R, %, with 95% bootstrap CI and n)"])
    w.writerow(["condition", "note"] + [x for _, tn in TRK for x in (f"{tn} R", f"{tn} 95% CI", f"{tn} n")] + ["Δ R (SAM 3.1 − SAM 3)"])
    for key, lab, note in B_ROWS:
        cells = []
        for t, _ in TRK:
            r, ci, n = bcell(key, t)
            cells += [f"{r:.1f}", f"{ci[0]:.1f}–{ci[1]:.1f}", n]
        w.writerow([lab, note] + cells + [f"{bcell(key, 'SAM3.1')[0] - bcell(key, 'SAM3')[0]:+.1f}"])
    w.writerow([])
    w.writerow(["Table C. Outcomes of occlusion events (counts)"])
    w.writerow(["outcome"] + [tn for _, tn in TRK])
    for k, lab in (("kept", "same ID (kept)"), ("switched", "new ID (switched)"), ("lost", "not found within 2 s (lost)"),
                   ("not_tracked_before", "not tracked before the event (excluded)")):
        w.writerow([lab] + [A[f"{t} {k}"] for t, _ in TRK])
    w.writerow([f"Paired SAM 3 vs SAM 3.1 on {PR['events_tracked_before_by_both']} events: only SAM 3 kept {PR['only_SAM3_kept']}, "
                f"only SAM 3.1 kept {PR['only_SAM3.1_kept']}, McNemar exact p = {PR['mcnemar_exact_p']:.1e}"])
    w.writerow([])
    w.writerow(["Table D. Per sequence, MOTS20 train"])
    w.writerow(["sequence", "resolution", "frames", "fps", "people", "visible people / frame", "% person-frames occluded"]
               + [f"AssA {tn}" for _, tn in TRK] + [f"R {tn}" for _, tn in TRK])
    for q in SEQS:
        sc = scene[q]
        w.writerow([q, sc["resolution"], sc["frames"], f"{sc['fps']:.0f}", sc["gt_people"], sc["visible_people_per_frame"],
                    sc["pct_person_frames_occluded"]] + [f"{DT[t][q]['AssA___AUC'] * 100:.1f}" for t, _ in TRK]
                   + [f"{rec(q, t):.1f}" for t, _ in TRK])
    w.writerow([])
    w.writerow(["Findings"])
    for k, txt in FINDINGS:
        w.writerow([k, txt])

# ------------------------------------------------------------------ HTML tables
E = html.escape


def dcls(x, better):
    if abs(x) < 1e-9 or better == "→1":
        return "d"
    worse = x < 0 if better == "↑" else x > 0
    return "d bad" if worse else "d good"


def tcls(t):
    return f"t-{t.replace('.', '')}"


def table_a():
    head1 = ("<tr><th rowspan='2' class='l'>Benchmark</th><th rowspan='2' class='l'>Tracker</th>"
             "<th colspan='4' class='grp'>HOTA and its components</th><th colspan='6' class='grp'>Identity errors</th></tr>")
    head2 = "<tr>" + "".join(f"<th class='{'key' if m in A_KEY else ''}'>{E(lab)}<span class='arr'>{a}</span></th>"
                             for m, lab, _, a in A_COLS) + "</tr>"
    rows = []
    for b, c, bl, cl in BENCH:
        for i, (t, tn) in enumerate(TRK):
            lead = f"<th rowspan='4' class='l bench'>{E(bl)}<span>{E(cl)}</span></th>" if i == 0 else ""
            cells = "".join(f"<td class='{'key' if m in A_KEY else ''}'>{fmt(v(b, c, t, m), d)}</td>" for m, _, d, _ in A_COLS)
            rows.append(f"<tr class='{tcls(t)}'>{lead}<th class='l trk'><i></i>{E(tn)}</th>{cells}</tr>")
        cells = "".join(
            f"<td class='{dcls(v(b, c, 'SAM3.1', m) - v(b, c, 'SAM3', m), a)}{' key' if m in A_KEY else ''}'>"
            f"{fdelta(v(b, c, 'SAM3.1', m) - v(b, c, 'SAM3', m), d)}</td>" for m, _, d, a in A_COLS)
        rows.append(f"<tr class='delta'><th class='l trk'>Δ 3.1 − 3</th>{cells}</tr>")
    return f"<table class='ta'><thead>{head1}{head2}</thead><tbody>{''.join(rows)}</tbody></table>"


def table_b():
    head = ("<tr><th class='l'>Condition</th>" + "".join(f"<th class='{tcls(t)}'><i></i>{E(tn)}</th>" for t, tn in TRK)
            + "<th>Δ 3.1 − 3</th></tr>")
    rows = []
    for key, lab, note in B_ROWS:
        cells = []
        for t, _ in TRK:
            r, ci, n = bcell(key, t)
            cells.append(f"<td><b>{r:.1f}</b><span class='ci'>[{ci[0]:.1f}, {ci[1]:.1f}] · n = {n}</span></td>")
        d = bcell(key, "SAM3.1")[0] - bcell(key, "SAM3")[0]
        cls = "sep" if key in ("all events", "duration <0.5 s", "depth partial (min vis 0.10-0.25)") else ""
        sub = " sub" if key.startswith("duration") and key != "duration <0.5 s" else ""
        note_html = f"<span>{E(note)}</span>" if note else ""
        rows.append(f"<tr class='{cls}{sub}'><th class='l'>{E(lab)}{note_html}</th>{''.join(cells)}"
                    f"<td class='{dcls(d, '↑')}'>{d:+.1f}</td></tr>")
    return f"<table class='tb'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def table_c():
    head = "<tr><th class='l'>Outcome of an event</th>" + "".join(f"<th class='{tcls(t)}'><i></i>{E(tn)}</th>" for t, tn in TRK) + "</tr>"
    rows = []
    for k, lab in (("kept", "Same ID (kept)"), ("switched", "New ID (switched)"), ("lost", "Not found within 2 s (lost)"),
                   ("not_tracked_before", "Not tracked before (excluded)")):
        rows.append(f"<tr class='{'sep' if k == 'not_tracked_before' else ''}'><th class='l'>{E(lab)}</th>"
                    + "".join(f"<td>{A[f'{t} {k}']}</td>" for t, _ in TRK) + "</tr>")
    return f"<table class='tc'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def table_d():
    head1 = ("<tr><th rowspan='2' class='l'>Sequence</th><th colspan='4' class='grp'>Scene</th>"
             "<th colspan='3' class='grp'>AssA</th><th colspan='3' class='grp'>R, identity kept through occlusion</th></tr>")
    tn3 = "".join(f"<th class='{tcls(t)}'><i></i>{E(tn)}</th>" for t, tn in TRK)
    head2 = f"<tr><th>frames · fps</th><th>people</th><th>visible / frame</th><th>% occluded</th>{tn3}{tn3}</tr>"
    rows = []
    for q in SEQS:
        sc = scene[q]
        cells = (f"<td>{sc['frames']} · {sc['fps']:.0f}</td><td>{sc['gt_people']}</td><td>{sc['visible_people_per_frame']:.1f}</td>"
                 f"<td>{sc['pct_person_frames_occluded']:.1f}</td>"
                 + "".join(f"<td>{DT[t][q]['AssA___AUC'] * 100:.1f}</td>" for t, _ in TRK)
                 + "".join(f"<td>{rec(q, t):.1f}</td>" for t, _ in TRK))
        rows.append(f"<tr class='{'hl' if q == worst else ''}'><th class='l'>{q}<span>{E(sc['resolution'])}</span></th>{cells}</tr>")
    return f"<table class='td'><thead>{head1}{head2}</thead><tbody>{''.join(rows)}</tbody></table>"


# ------------------------------------------------------------------ notation (MathML: native in current browsers, prints as text)
def mi(x):
    return f"<mi>{x}</mi>"


def mt(x):
    return f"<mtext>{x}</mtext>"


def sub(base, s):
    return f"<msub>{base}{s}</msub>"


def card(x):                                  # |x| as a set cardinality: no operator spacing around the bars
    bar = "<mo stretchy='false' lspace='0' rspace='0'>|</mo>"
    return f"<mrow>{bar}{x}{bar}</mrow>"


def frac(a, b):
    return f"<mfrac><mrow>{a}</mrow><mrow>{b}</mrow></mfrac>"


al = mi("α")
TPa, FNa, FPa = sub(mt("TP"), al), sub(mt("FN"), al), sub(mt("FP"), al)
TPA, FNA, FPA = (f"{mt(x)}<mo>(</mo>{mi('c')}<mo>)</mo>" for x in ("TPA", "FNA", "FPA"))
SUM = "<mo largeop='true' movablelimits='false' style='font-size:1.45em'>∑</mo>"
sum_c = f"<munder>{SUM}<mrow>{mi('c')}<mo>∈</mo>{TPa}</mrow></munder>"
inv = frac("<mn>1</mn>", card(TPa))
PLUS = "<mo>+</mo>"
MATH = [
    ("HOTA",
     f"{sub(mt('HOTA'), al)}<mo>=</mo><msqrt>{sub(mt('DetA'), al)}<mo>⋅</mo>{sub(mt('AssA'), al)}</msqrt><mo>,</mo><mspace width='1em'/>"
     f"{mt('HOTA')}<mo>=</mo>{frac('<mn>1</mn>', card(mi('A')))}<munder>{SUM}<mrow>{al}<mo>∈</mo>{mi('A')}</mrow></munder>{sub(mt('HOTA'), al)}"
     "",
     "A = {0.05, 0.10, …, 0.95} (|A| = 19). At each localisation threshold α, a predicted mask matches a ground-truth mask if their IoU ≥ α (one-to-one). DetA, AssA, AssRe, "
     "AssPr and LocA are averaged over the same 19 thresholds, so the reported HOTA is not √(DetA · AssA) of the averaged values."),
    ("DetA", f"{sub(mt('DetA'), al)}<mo>=</mo>{frac(card(TPa), card(TPa) + PLUS + card(FNa) + PLUS + card(FPa))}",
     "Detection accuracy: matched masks over all ground-truth and predicted masks."),
    ("AssA", f"{sub(mt('AssA'), al)}<mo>=</mo>{inv}{sum_c}{frac(card(TPA), card(TPA) + PLUS + card(FNA) + PLUS + card(FPA))}",
     "For a match c between ground-truth object g and predicted identity p: TPA(c) = matches with the same pair (g, p); "
     "FNA(c) = frames in which g is present but not matched to p; FPA(c) = frames in which p is present but not matched to g."),
    ("AssRe", f"{sub(mt('AssRe'), al)}<mo>=</mo>{inv}{sum_c}{frac(card(TPA), card(TPA) + PLUS + card(FNA))}",
     "Association recall. Low AssRe: one object is split over several identities."),
    ("AssPr", f"{sub(mt('AssPr'), al)}<mo>=</mo>{inv}{sum_c}{frac(card(TPA), card(TPA) + PLUS + card(FPA))}",
     "Association precision. Low AssPr: one identity covers several objects (a merge)."),
    ("LocA", f"{sub(mt('LocA'), al)}<mo>=</mo>{inv}{sum_c}{mt('IoU')}<mo>(</mo>{mi('c')}<mo>)</mo>",
     "Mean mask IoU of matched pairs: segmentation quality."),
    ("R and Δ",
     f"{mi('R')}<mo>=</mo>{frac(sub(mi('N'), mt('kept')), sub(mi('N'), mt('kept')) + PLUS + sub(mi('N'), mt('switched')) + PLUS + sub(mi('N'), mt('lost')))}"
     f"<mo>,</mo><mspace width='1em'/><mi mathvariant='normal'>Δ</mi><mo>=</mo>{sub(mi('x'), mt('SAM 3.1'))}<mo>−</mo>{sub(mi('x'), mt('SAM 3'))}",
     "R: identity recovery over occlusion events the tracker was following just before (matched within 1 s before; mask IoU ≥ 0.5). "
     "Kept: same ID at the first match within 2 s after; switched: a different ID; lost: no match within 2 s. Δ: difference of any metric x."),
]
math_html = "".join(f"<div class='eq'><div class='eqn'><math display='block'><mstyle displaystyle='true'>{m}</mstyle></math></div><p><b>{E(k)}.</b> {E(t)}</p></div>"
                    for k, m, t in MATH)

chart = base64.b64encode((OCC / "id_recovery_by_duration.png").read_bytes()).decode()
example = base64.b64encode((OCC / "example_MOTS20-02_person26.jpg").read_bytes()).decode()
findings_html = "".join(f"<li><b>{E(k)}</b> {E(t)}</li>" for k, t in FINDINGS)

page = f"""<title>SAM Occlusion Diagnosis</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500&display=swap">
<style>
/* Layout: one reading column; findings, then four tables narrowing from benchmark metrics to occlusion to scenes, then notation. */
:root {{
  --bg: #f5f6f7; --paper: #ffffff; --ink: #1c2025; --muted: #5a626c; --rule: #dcdfe4; --head: #eef0f3;
  --sam3: #2a78d6; --sam31: #d45b1f; --ref: #8a8a84; --bad: #b3261e; --bad-bg: #fcebe9; --good: #1f7a4a; --hl: #fff5e0;
  --serif: "IBM Plex Serif", Georgia, serif; --sans: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #14171b; --paper: #1c2025; --ink: #e7e9ec; --muted: #9aa2ac; --rule: #333941; --head: #242930;
  --sam3: #6aa6ef; --sam31: #f08a55; --ref: #a9a9a2; --bad: #f28b82; --bad-bg: #3b2221; --good: #6fcf97; --hl: #2d2718; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #14171b; --paper: #1c2025; --ink: #e7e9ec; --muted: #9aa2ac; --rule: #333941; --head: #242930;
  --sam3: #6aa6ef; --sam31: #f08a55; --ref: #a9a9a2; --bad: #f28b82; --bad-bg: #3b2221; --good: #6fcf97; --hl: #2d2718; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 14.5px/1.55 var(--sans); }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding-inline: 20px; padding-block: 34px 60px; display: grid; gap: 30px; }}
header {{ display: grid; gap: 8px; }}
.eyebrow {{ font: 500 11.5px/1.2 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
h1 {{ font: 500 clamp(25px, 3.2vw, 34px)/1.15 var(--serif); margin: 0; text-wrap: balance; }}
.lede {{ margin: 0; color: var(--muted); max-width: 88ch; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 12.5px; color: var(--muted); }}
.legend > span {{ display: inline-flex; align-items: center; gap: 6px; }}
.sw {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; }}
i {{ display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 6px; }}
.t-SAM3 i, .sw3 {{ background: var(--sam3); }} .t-SAM31 i, .sw31 {{ background: var(--sam31); }} .t-track_rcnn i, .swr {{ background: var(--ref); }}
section {{ display: grid; gap: 10px; min-width: 0; }}
h2 {{ margin: 0; font: 600 17px/1.3 var(--sans); text-wrap: balance; }}
h2 small {{ font: 400 13px var(--sans); color: var(--muted); margin-left: 6px; }}
.findings ol {{ margin: 0; padding-left: 22px; display: grid; gap: 7px; max-width: 96ch; font-variant-numeric: tabular-nums; }}
.findings b {{ font-weight: 600; }}
.box {{ overflow-x: auto; background: var(--paper); border: 1px solid var(--rule); border-radius: 5px; }}
table {{ border-collapse: collapse; width: 100%; font: 13px/1.35 var(--mono); font-variant-numeric: tabular-nums; }}
th, td {{ padding: 6px 9px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--rule); }}
thead th {{ background: var(--head); font: 500 11.5px/1.3 var(--mono); color: var(--muted); vertical-align: bottom; }}
thead th.grp {{ text-align: center; color: var(--ink); border-left: 1px solid var(--rule); }}
th.l {{ text-align: left; }}
tbody th {{ font: 500 12.5px/1.3 var(--sans); }}
tbody th span {{ display: block; font: 400 11.5px/1.3 var(--sans); color: var(--muted); }}
.arr {{ display: block; font-size: 10.5px; color: var(--muted); }}
thead th.key {{ color: var(--ink); }}
td.key {{ font-weight: 500; }}
.ta tbody th.bench {{ vertical-align: top; border-right: 1px solid var(--rule); }}
.ta tr.delta > * {{ background: var(--head); font-size: 12px; border-bottom: 2px solid var(--rule); }}
.ta tr.delta th {{ font: 500 11.5px var(--mono); color: var(--muted); }}
.ta tr.delta td.d.bad {{ color: var(--bad); background: var(--bad-bg); }}
td.d.bad {{ color: var(--bad); background: var(--bad-bg); }}
td.d.good {{ color: var(--good); }}
.tb td b {{ font-weight: 600; font-size: 13.5px; }}
.ci {{ display: block; font-size: 11px; color: var(--muted); }}
tr.sep > * {{ border-top: 2px solid var(--rule); }}
tr.sub th {{ padding-left: 22px; }}
tr.hl > * {{ background: var(--hl); }}
.cap {{ margin: 0; color: var(--muted); font-size: 13px; max-width: 100ch; }}
.cap b {{ color: var(--ink); font-weight: 600; }}
.pair {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr); gap: 20px; align-items: start; }}
@media (max-width: 820px) {{ .pair {{ grid-template-columns: 1fr; }} }}
figure {{ margin: 0; display: grid; gap: 6px; min-width: 0; }}
figure img {{ width: 100%; border: 1px solid var(--rule); border-radius: 5px; background: #fff; }}
figcaption {{ font-size: 12.5px; color: var(--muted); }}
.eqs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 470px), 1fr)); gap: 14px 28px; }}
.eq {{ display: grid; gap: 2px; min-width: 0; }}
.eqn {{ overflow-x: auto; }}
math {{ font-size: 1.08em; }}
.eq p {{ margin: 0; font-size: 12.5px; color: var(--muted); }}
.eq p b {{ color: var(--ink); }}
.meth ul {{ margin: 0; padding-left: 20px; display: grid; gap: 4px; font-size: 13px; color: var(--muted); max-width: 100ch; }}
code {{ font: 12px var(--mono); }}
@media print {{
  @page {{ size: A4 landscape; margin: 10mm; }}
  :root, :root[data-theme="dark"] {{ --bg: #fff; --paper: #fff; --ink: #000; --muted: #444; --rule: #bbb; --head: #f0f0f0;
    --sam3: #2a78d6; --sam31: #d45b1f; --ref: #8a8a84; --bad: #a11; --bad-bg: #fbe9e7; --good: #176; --hl: #fff4dc; color-scheme: light; }}
  body {{ font-size: 10pt; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  .wrap {{ max-width: none; padding: 0; gap: 14px; }}
  .box {{ overflow: visible; }}
  table {{ font-size: 8.4pt; }} th, td {{ padding: 3px 5px; }}
  section, figure, .eq, tr {{ break-inside: avoid; }}
  .pb {{ break-before: page; }}
  .pair {{ grid-template-columns: 1fr 1.25fr; }}
  figure.ex img {{ max-height: 95mm; width: auto; max-width: 100%; }}
}}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">Occlusion study · diagnosis before solutions · 30 Sep 2026</div>
  <h1>Where SAM loses identity under occlusion</h1>
  <p class="lede">SAM 3 and SAM 3.1, zero-shot with the class name as the only prompt, against the benchmark reference TrackR-CNN
  (Voigtlaender et al., CVPR 2019). Official TrackEval (HOTA, CLEAR, Identity) on KITTI-MOTS val (9 sequences) and MOTS20 train
  (4 sequences); occlusion events from MOT17's per-frame visibility labels for the same videos (245 events, 204 pedestrians).</p>
  <div class="legend"><span><span class="sw sw3"></span>SAM 3</span><span><span class="sw sw31"></span>SAM 3.1</span>
  <span><span class="sw swr"></span>TrackR-CNN (reference)</span><span>Δ = SAM 3.1 − SAM 3; red where SAM 3.1 is worse</span>
  <span>↑ higher is better · ↓ lower is better</span><span>scores in %, except counts and ratios</span></div>
</header>

<section class="findings"><h2>Findings</h2><ol>{findings_html}</ol></section>

<section>
  <h2>A. Benchmark results <small>TrackEval, combined over sequences</small></h2>
  <div class="box">{table_a()}</div>
  <p class="cap"><b>Reading.</b> AssA (association) is the component with the largest SAM 3.1 − SAM 3 drop on every benchmark, while
  LocA (mask quality) is unchanged. On MOTS20 the drop is in AssRe (splits), not AssPr (merges). IDSW, Frag and ML are CLEAR-MOT
  quantities at mask IoU ≥ 0.5; ML % = share of people tracked for less than 20% of their time in view; IDs / GT IDs = predicted
  identities per person (ideal 1).</p>
</section>

<section class="pb">
  <h2>B. Identity kept through occlusion <small>MOTS20 train · R in %, [95% CI] · n</small></h2>
  <div class="box">{table_b()}</div>
  <p class="cap"><b>Reading.</b> Without occlusion all three trackers keep identities {min(ctrl.values()):.0f}–{max(ctrl.values()):.0f}% of the time;
  through occlusion they separate sharply, and each is lowest for occlusions of 2 s or more. CI: percentile bootstrap over people, 10<sup>4</sup> resamples.</p>
</section>

<div class="pair">
<section>
  <h2>C. What happens to the identity <small>all events</small></h2>
  <div class="box">{table_c()}</div>
  <p class="cap">Paired, SAM 3 vs SAM 3.1 on the {PR['events_tracked_before_by_both']} events both followed beforehand: only SAM 3 kept the ID
  {PR['only_SAM3_kept']} times, only SAM 3.1 {PR['only_SAM3.1_kept']} time; exact McNemar test p = {PR['mcnemar_exact_p']:.1e}.</p>
</section>
<figure><img alt="Bar chart of identity recovery by occlusion duration for SAM 3, SAM 3.1 and TrackR-CNN with 95% confidence intervals" src="data:image/png;base64,{chart}">
<figcaption>R by occlusion duration; the left group is the no-occlusion control.</figcaption></figure>
</div>

<section>
  <h2>D. Per sequence <small>MOTS20 train · highlighted: largest SAM 3.1 − SAM 3 gap in R</small></h2>
  <div class="box">{table_d()}</div>
  <p class="cap"><b>Reading.</b> SAM 3.1 keeps fewer identities than SAM 3 on every sequence. "% occluded": share of person-frames with
  MOT17 visibility &lt; 0.25; "visible / frame": people with a ground-truth mask per frame.</p>
</section>

<figure class="ex"><img alt="MOTS20-02, person 26 before, during and after an occlusion: SAM 3 keeps ID 14, SAM 3.1 changes ID 9 to 83" src="data:image/jpeg;base64,{example}">
<figcaption>Checked by hand: MOTS20-02, person 26 (white box), hidden for 0.5 s. SAM 3 keeps ID 14; SAM 3.1 changes ID 9 → 83.</figcaption></figure>

<section class="pb">
  <h2>Notation <small>as computed by TrackEval (Luiten et al., IJCV 2021) and by our occlusion analysis</small></h2>
  <div class="eqs">{math_html}</div>
</section>

<section class="meth">
  <h2>Method and limits</h2>
  <ul>
    <li>Models run as in Meta's official notebooks (text prompt on frame 0, then propagation); SAM 3.1 with <code>max_num_objects=128</code>, its model default.
        GPU-memory offloads were verified to give byte-identical outputs. TrackR-CNN: the official outputs shipped with TrackEval, which reproduce its published sMOTSA.</li>
    <li>Occlusion event: MOT17 visibility &lt; 0.25 on consecutive frames, with the person visible (≥ 0.25) just before and after; duration = frames / fps.
        Thresholds were fixed before results; TrackR-CNN was added afterwards with the same definitions.</li>
    <li>Pedestrians only for B–D (KITTI has no visibility labels). 24 of 228 people could not be linked to a MOT17 track and are excluded;
        events a tracker was not following beforehand are excluded from its R (Table C). The ≥ 2 s bin is small (n ≈ 25), hence wide intervals.</li>
    <li>Open question for the solutions phase: SAM 3.1's multiplexed memory under crowding (its log reports bucket subscription of 1350% on MOTS20-02).</li>
  </ul>
</section>
</div>
"""
(HERE / "occlusion_diagnosis.html").write_text(page)
print("wrote", out_csv.name, "occlusion_diagnosis.html")
