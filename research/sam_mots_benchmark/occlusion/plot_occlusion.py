"""Chart for occlusion_results.json: ID recovery after occlusion, by duration, with 95% CIs and the no-occlusion control."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
COL = {"SAM3": "#2a78d6", "SAM3.1": "#eb6834", "track_rcnn": "#898781"}
NAME = {"SAM3": "SAM 3", "SAM3.1": "SAM 3.1", "track_rcnn": "TrackR-CNN (reference)"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
by = {s["subset"]: s for s in res["summary"]}
groups = [("no occlusion\n(control, 1 s)", None), ("all\nocclusions", "all events"),
          ("< 0.5 s", "duration <0.5 s"), ("0.5–1 s", "duration 0.5-1 s"), ("1–2 s", "duration 1-2 s"),
          ("≥ 2 s", "duration >=2 s")]
fig, ax = plt.subplots(figsize=(9.2, 4.8))
x = np.arange(len(groups))
counts = {}
for k, (t, dx) in enumerate((("SAM3", -0.27), ("SAM3.1", 0.0), ("track_rcnn", 0.27))):
    vals, lo, hi, ns = [], [], [], []
    for _, key in groups:
        if key is None:
            c = res["control_no_occlusion"][t]
            v, ci, n = c["same_id_rate"], c["ci"], c["windows"]
        else:
            s = by[key]
            v, ci, n = s[f"{t} recovery"], s[f"{t} recovery_ci"], s[f"{t} events"]
        vals.append(v * 100); lo.append((v - ci[0]) * 100); hi.append((ci[1] - v) * 100); ns.append(n)
    counts[t] = ns
    ax.bar(x + dx, vals, 0.26, color=COL[t], label=NAME[t], zorder=2)
    ax.errorbar(x + dx, vals, yerr=[lo, hi], fmt="none", ecolor="#3b3b3b", elinewidth=1, capsize=3, zorder=3)
    for xi, v, n in zip(x + dx, vals, ns):
        inside = v >= 10
        ax.text(xi, 2 if inside else v + 1, f"{v:.0f}", ha="center", va="bottom", color="white" if inside else "#3b3b3b",
                fontsize=7.5, fontweight="bold", zorder=4)
ax.axvline(0.5, color="#c9c7c0", lw=1, ls="--")
ax.set_xticks(x, [f"{g}\nn = {a} / {b} / {c}" for (g, _), a, b, c in zip(groups, counts["SAM3"], counts["SAM3.1"], counts["track_rcnn"])],
              fontsize=8.5)
ax.set_ylim(0, 105)
ax.set_ylabel("same identity after (%)")
ax.grid(axis="y", color="#e6e4dd", zorder=0)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
ax.set_title("Same identity after an occlusion? (MOTS20 train; occlusion = MOT17 visibility < 0.25)",
             loc="left", fontsize=10.5, fontweight="bold")
fig.text(0.01, 0.005, "Bars: ID recovery rate (same ID after the person reappears / all events tracked before). "
         "Whiskers: 95% bootstrap CI over people. n = SAM 3 / SAM 3.1 / TrackR-CNN.", fontsize=7.5, color="#6b6b6b")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(out, dpi=160)
