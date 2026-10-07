#!/usr/bin/env python3
"""Build the results report from TrackEval outputs: tables (CSV), charts (PNG) and a PDF report.

Every number is parsed from TrackEval's own output files (<class>_summary.txt,
<class>_detailed.csv) and the runner's run_info.json; nothing is typed by hand.

    python build_report.py --results results --trackers trackers --out report \
        --narrative narrative.md
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TRACKERS = [("SAM3", "SAM 3"), ("SAM3.1", "SAM 3.1"), ("track_rcnn", "TrackR-CNN (reference)")]
COLORS = {"SAM3": "#2a78d6", "SAM3.1": "#eb6834", "track_rcnn": "#898781"}      # validated light palette
GROUPS = [("kitti_mots", "car", "KITTI-MOTS val · car"),
          ("kitti_mots", "pedestrian", "KITTI-MOTS val · pedestrian"),
          ("mots_challenge", "pedestrian", "MOTS20 train · pedestrian")]
ASSOC = ["AssA", "AssRe", "AssPr", "IDF1", "IDSW"]
MAIN = ["HOTA", "DetA", "AssA", "AssRe", "AssPr", "LocA", "IDF1", "IDSW", "Frag", "sMOTA", "MOTA", "IDs", "GT_IDs"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 160, "savefig.bbox": "tight",
})


def read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    head, vals = path.read_text().splitlines()[:2]
    return {k: float(v) for k, v in zip(head.split(), vals.split())}


def read_detailed(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return {row["seq"]: {k: float(v) for k, v in row.items() if k != "seq"} for row in csv.DictReader(f)}


def load(results: Path) -> dict:
    data = {}
    for bench, cls, _ in GROUPS:
        for tid, _ in TRACKERS:
            base = results / bench / tid
            s = read_summary(base / f"{cls}_summary.txt")
            d = read_detailed(base / f"{cls}_detailed.csv")
            if s:
                data[(bench, cls, tid)] = {"summary": s, "detailed": d}
    return data


def fmt(k: str, v: float) -> str:
    return f"{int(round(v))}" if k in ("IDSW", "Frag", "IDs", "GT_IDs") else f"{v:.1f}"


# ------------------------------------------------------------------ charts
def panel_chart(data, panels: list[tuple[str, str]], title: str, out: Path, ncols: int = 2):
    """Small multiples: one panel per metric, grouped bars per benchmark/class."""
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.4, 2.35 * nrows + 0.5), squeeze=False)
    for ax, (metric, sub) in zip(axes.flat, panels):
        _bars(ax, data, metric)
        ax.set_title(f"{metric} · {sub}", loc="left", fontsize=9.5, color=INK, fontweight="bold")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    present = [t for t in TRACKERS if any((b, c, t[0]) in data for b, c, _ in GROUPS)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[t]) for t, _ in present]
    fig.legend(handles, [n for _, n in present], frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.01, -0.02),
               fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def _bars(ax, data, metric: str):
    x = np.arange(len(GROUPS))
    present = [t for t in TRACKERS if any((b, c, t[0]) in data for b, c, _ in GROUPS)]
    width = 0.8 / max(len(present), 1)
    lo = metric in ("IDSW", "Frag")
    top = 0
    for k, (tid, name) in enumerate(present):
        vals = [data.get((b, c, tid), {}).get("summary", {}).get(metric, np.nan) for b, c, _ in GROUPS]
        pos = x + (k - (len(present) - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.92, color=COLORS[tid], edgecolor="white", linewidth=1)
        top = max(top, np.nanmax(vals) if np.isfinite(vals).any() else 0)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}" if lo else f"{v:.1f}", ha="center",
                        va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, [g[2].replace(" · ", "\n") for g in GROUPS], fontsize=7.5)
    ax.set_ylim(0, top * 1.15 if lo else 100)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="y", labelsize=7.5)


def bar_chart(data, metric: str, title: str, out: Path, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    x = np.arange(len(GROUPS))
    width = 0.26
    present = [t for t in TRACKERS if any((b, c, t[0]) in data for b, c, _ in GROUPS)]
    for k, (tid, name) in enumerate(present):
        vals = [data.get((b, c, tid), {}).get("summary", {}).get(metric, np.nan) for b, c, _ in GROUPS]
        pos = x + (k - (len(present) - 1) / 2) * (width + 0.02)
        bars = ax.bar(pos, vals, width, color=COLORS[tid], label=name, edgecolor="white", linewidth=1.5)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5,
                        color=INK)
    ax.set_xticks(x, [g[2] for g in GROUPS])
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def alpha_chart(data, metric: str, out: Path):
    alphas = np.arange(5, 100, 5)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    for ax, (bench, cls, label) in zip(axes, GROUPS):
        for tid, name in TRACKERS:
            d = data.get((bench, cls, tid), {}).get("detailed")
            if not d:
                continue
            ys = [d["COMBINED"][f"{metric}___{a}"] * 100 for a in alphas]
            ax.plot(alphas / 100, ys, color=COLORS[tid], lw=2, label=name)
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("localisation threshold α", fontsize=8.5)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel(f"{metric} (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"{metric} across localisation thresholds (HOTA's α sweep)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def per_sequence_chart(data, metric: str, out: Path):
    label_metric, metric = metric, (metric + "___AUC" if metric in ("HOTA", "DetA", "AssA") else metric)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [9, 9, 4]}, sharey=True)
    for ax, (bench, cls, label) in zip(axes, GROUPS):
        d3 = data.get((bench, cls, "SAM3"), {}).get("detailed") or {}
        d31 = data.get((bench, cls, "SAM3.1"), {}).get("detailed") or {}
        seqs = [s for s in d3 if s != "COMBINED" and d3[s].get("GT_Dets", 1) > 0]   # skip sequences without this class
        xs = np.arange(len(seqs))
        for tid, d, dx in (("SAM3", d3, -0.12), ("SAM3.1", d31, 0.12)):
            ys = [d.get(s, {}).get(metric, np.nan) * 100 if d.get(s) else np.nan for s in seqs]
            ax.scatter(xs + dx, ys, s=28, color=COLORS[tid], zorder=3, edgecolor="white", linewidth=1)
        for i, s in enumerate(seqs):
            a, b = d3.get(s, {}).get(metric), d31.get(s, {}).get(metric)
            if a is not None and b is not None:
                ax.plot([i - 0.12, i + 0.12], [a * 100, b * 100], color=GRID, lw=1.5, zorder=2)
        ax.set_xticks(xs, [s.replace("MOTS20-", "") for s in seqs], fontsize=7.5, rotation=0)
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(f"{label_metric} (%)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[t], label=n) for t, n in TRACKERS[:2]]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"{label_metric} per sequence (a line joins the two models on the same sequence)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ tables / csv
def write_csvs(data, runs, out: Path):
    with open(out / "results_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "split", "class", "tracker"] + MAIN + ["ms_per_frame"])
        for bench, cls, _ in GROUPS:
            for tid, name in TRACKERS:
                e = data.get((bench, cls, tid))
                if not e:
                    continue
                s = e["summary"]
                w.writerow([bench, "val" if bench == "kitti_mots" else "train", cls, name]
                           + [fmt(k, s.get(k, float("nan"))) for k in MAIN]
                           + [runs.get((bench, tid), {}).get("ms_per_frame", "")])
    with open(out / "results_per_sequence.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = ["HOTA", "DetA", "AssA", "AssRe", "AssPr", "LocA", "IDF1", "IDSW"]
        w.writerow(["benchmark", "class", "tracker", "sequence"] + keys)
        for bench, cls, _ in GROUPS:
            for tid, name in TRACKERS:
                d = (data.get((bench, cls, tid)) or {}).get("detailed") or {}
                for seq, row in d.items():
                    if row.get("GT_Dets", 1) == 0:          # sequence has no objects of this class
                        continue
                    col = {k: (k + "___AUC" if k + "___AUC" in row else k) for k in keys}   # HOTA family: AUC over alpha
                    vals = [fmt(k, row[col[k]] * (1 if k in ("IDSW",) else 100)) if col[k] in row else "" for k in keys]
                    w.writerow([bench, cls, name, seq] + vals)


def html_table(data, runs, keys, bench, cls) -> str:
    head = "".join(f"<th>{k}</th>" for k in keys) + "<th>ms / frame</th>"
    rows = []
    best = {}
    for k in keys:
        vals = [(data.get((bench, cls, t)) or {}).get("summary", {}).get(k) for t, _ in TRACKERS[:2]]
        vals = [v for v in vals if v is not None]
        if vals:
            best[k] = min(vals) if k in ("IDSW", "Frag") else max(vals)
    for tid, name in TRACKERS:
        e = data.get((bench, cls, tid))
        if not e:
            continue
        s = e["summary"]
        cells = []
        for k in keys:
            v = s.get(k)
            cls_attr = ' class="best"' if tid != "track_rcnn" and v is not None and v == best.get(k) else ""
            cells.append(f"<td{cls_attr}>{fmt(k, v) if v is not None else '–'}</td>")
        ms = runs.get((bench, tid), {}).get("ms_per_frame")
        cells.append(f"<td>{ms:.0f}</td>" if ms else "<td>–</td>")
        dot = f'<span class="dot" style="background:{COLORS[tid]}"></span>'
        rows.append(f"<tr class='{'ref' if tid == 'track_rcnn' else ''}'><th>{dot}{html.escape(name)}</th>{''.join(cells)}</tr>")
    return f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def load_runs(trackers: Path) -> dict:
    """Merge the per-worker run_info*.json of each (benchmark, tracker): per sequence, the successful run."""
    runs = {}
    for tdir in sorted({p.parent for p in trackers.glob("*/*/run_info*.json")}):
        seqs, errors, meta = {}, set(), None
        for info in sorted(tdir.glob("run_info*.json")):
            r = json.loads(info.read_text())
            meta = meta or r
            for k, v in r["sequences"].items():
                if "frames" in v:
                    seqs[k] = v
                else:
                    errors.add(k)
        frames = sum(v["frames"] for v in seqs.values())
        secs = sum(v["seconds"] for v in seqs.values())
        runs[(meta["benchmark"], meta["tracker_name"])] = {
            "ms_per_frame": 1000 * secs / frames if frames else None, "frames": frames, "seconds": secs,
            "gpu": meta.get("gpu"), "errors": sorted(errors - set(seqs)), "sequences": sorted(seqs),
            "dropped": sum(v.get("dropped_after_overlap", 0) for v in seqs.values()),
            "max_objects": max((v["objects"] for v in seqs.values()), default=0)}
    return runs


def md_to_html(md: str) -> str:
    out, in_list = [], False
    for line in md.splitlines():
        line = html.escape(line)
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{line[2:]}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("### "):
            out.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("| "):
            out.append(line)
        elif line.strip():
            out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif; color: #0b0b0b; font-size: 10.2pt; line-height: 1.45; }
h1 { font-size: 19pt; margin: 0 0 2mm; } h2 { font-size: 13pt; margin: 7mm 0 2mm; border-bottom: 1px solid #e1e0d9; padding-bottom: 1mm; }
h3 { font-size: 11pt; margin: 4mm 0 1mm; } .sub { color: #52514e; margin-bottom: 5mm; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 2mm 0 4mm; font-variant-numeric: tabular-nums; }
th, td { padding: 1.4mm 1.6mm; text-align: right; border-bottom: 1px solid #e1e0d9; } thead th { color: #52514e; font-weight: 600; border-bottom: 1.5px solid #c3c2b7; }
tbody th { text-align: left; font-weight: 600; white-space: nowrap; } td.best { font-weight: 700; }
tr.ref th, tr.ref td { color: #52514e; } .dot { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 5px; }
img { width: 100%; margin: 1mm 0 3mm; } .note { color: #52514e; font-size: 8.8pt; } code { font-size: 8.8pt; background: #f0efec; padding: 0 2px; border-radius: 2px; }
.pb { page-break-before: always; } .keep { page-break-inside: avoid; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--trackers", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, default=Path(__file__).with_name("PROTOCOL.md"))
    ap.add_argument("--narrative", type=Path, help="markdown: executive summary + findings")
    ap.add_argument("--date", default="")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig_dir = a.out / "figures"
    fig_dir.mkdir(exist_ok=True)
    data, runs = load(a.results), load_runs(a.trackers)

    panel_chart(data, [("AssA", "association accuracy"), ("IDF1", "identity F1"),
                       ("AssRe", "association recall: one object, one track"),
                       ("AssPr", "association precision: one track, one object")],
                "Association: does each object keep a single, correct identity?", fig_dir / "1_association.png")
    panel_chart(data, [("HOTA", "overall: mean over α of √(DetAα · AssAα)"), ("DetA", "detection accuracy"),
                       ("IDSW", "identity switches (lower is better)"), ("LocA", "mask localisation")],
                "Overall accuracy, detection, and identity switches", fig_dir / "2_overall.png")
    alpha_chart(data, "AssA", fig_dir / "5_assa_alpha.png")
    per_sequence_chart(data, "AssA", fig_dir / "6_assa_per_sequence.png")
    write_csvs(data, runs, a.out)

    narrative = md_to_html(a.narrative.read_text()) if a.narrative and a.narrative.exists() else ""
    tables = "".join(
        f"<div class='keep'><h3>{label}</h3>{html_table(data, runs, MAIN, b, c)}</div>" for b, c, label in GROUPS)
    body = f"""
<h1>SAM 3 vs SAM 3.1 · zero-shot multi-object tracking &amp; segmentation</h1>
<div class="sub">KITTI-MOTS (val) and MOTSChallenge MOTS20 (train) · official TrackEval HOTA · focus: association{(' · ' + a.date) if a.date else ''}</div>
{narrative}
<h2>Association</h2>
<p class="note">AssA (association accuracy) measures how consistently each ground-truth object is followed by one predicted
identity across its lifetime, over all correctly detected frames. It is the part of HOTA that occlusion stresses.
AssRe penalises one object being split into several tracks; AssPr penalises one track covering several objects.</p>
<img src="figures/1_association.png">
<div class="pb"></div>
<h2>Overall accuracy and its two halves</h2>
<p class="note">At each localisation threshold α (0.05 … 0.95), HOTAα is the geometric mean of detection (DetAα) and association
(AssAα) accuracy; HOTA is the mean of HOTAα over the 19 thresholds. A tracker can lose HOTA by finding fewer objects (DetA) or by breaking their identities (AssA).</p>
<img src="figures/2_overall.png">
<h2>Association in detail</h2>
<img src="figures/5_assa_alpha.png"><img src="figures/6_assa_per_sequence.png">
<div class="pb"></div>
<h2>Full results (TrackEval, combined over sequences)</h2>
<p class="note">Bold = better of SAM 3 / SAM 3.1. IDSW = identity switches, Frag = track fragmentations, IDs / GT_IDs =
number of predicted / ground-truth identities. sMOTA is TrackEval's name for sMOTSA. ms / frame = end-to-end inference time
on one A100 80GB including all prompts (several workers shared the GPU, so absolute times are indicative only).</p>
{tables}
<p class="note">Per-sequence values for every metric: <code>results_per_sequence.csv</code>. Raw TrackEval outputs are kept on the pod.</p>
<div class="pb"></div>
<h2>Protocol and reproducibility</h2>
{md_to_html(a.protocol.read_text().split(chr(10), 1)[1]) if a.protocol.exists() else ''}
"""
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>SAM 3 vs SAM 3.1 MOTS benchmark</title><style>{CSS}</style></head><body>{body}</body></html>"
    # protocol tables are markdown pipes: render them as preformatted text blocks
    page = re.sub(r"((?:\|[^\n]*\n?)+)", lambda m: "<pre class='note'>" + m.group(1) + "</pre>", page)
    (a.out / "report.html").write_text(page)
    if a.protocol.exists():
        shutil.copy(a.protocol, a.out / "PROTOCOL.md")
    print(f"built {a.out}: {len(data)} result sets, runs={ {k: round(v['ms_per_frame'] or 0) for k, v in runs.items()} }")
    for k, v in runs.items():
        if v["errors"]:
            print("  ERRORS in", k, v["errors"])


if __name__ == "__main__":
    main()
