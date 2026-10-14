#!/usr/bin/env python3
"""Development data that is not scored: the ground truth of the 12 KITTI-MOTS training sequences outside the val
split, with every object given a NEW id each time it comes back after a gap. That is the failure we want to repair
(found again, but under a new identity), on perfect masks, so the filter's settings can be checked without touching
the benchmark sequences.

    python dev_fragments.py <out tracker dir>
"""
import sys
from pathlib import Path

from ekf_relink import DATA, read

GT = DATA / "x_data/data/gt/kitti/kitti_mots_train"
out = Path(sys.argv[1]) / "data"
out.mkdir(parents=True, exist_ok=True)
gaps = 0
for line in (GT / "evaluate_mots.seqmap.training_minus_val").read_text().split("\n"):
    if not line.strip():
        continue
    seq = line.split()[0]
    last, part, rows = {}, {}, []
    for f, dets in sorted(read(GT / "label_02" / f"{seq}.txt").items()):
        for tid, cls, h, w, rle in dets:
            if tid in last and f - last[tid] > 1:
                part[tid] = part.get(tid, 0) + 1; gaps += 1
            last[tid] = f
            rows.append(f"{f} {tid * 100 + part.get(tid, 0)} {cls} {h} {w} {rle}")
    (out / f"{seq}.txt").write_text("\n".join(rows) + "\n")
print(f"{gaps} gaps turned into new ids")
