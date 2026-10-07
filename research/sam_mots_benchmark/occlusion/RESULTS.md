# Identity through occlusion: results

Method: `OCCLUSION_PROTOCOL.md` (fixed before running). Data: MOTS20 train (4 videos), with
occlusion taken from MOT17's per-frame visibility labels. 245 occlusion events across 204 linked
pedestrians.

![ID recovery by occlusion duration](results/id_recovery_by_duration.png)

| | SAM 3 | SAM 3.1 |
|---|---|---|
| **ID recovery after occlusion** (95% CI) | **93.6%** (89.2–97.2) | 63.3% (54.7–71.6) |
| same, occlusion ≥ 2 s | 73.1% | 36.0% |
| same, heavy/full occlusion (min visibility < 0.1) | 92.5% | 57.3% |
| outcomes: kept / switched / lost | 190 / 7 / 6 | 114 / 56 / 10 |
| **control: same ID after 1 s, no occlusion** | 98.1% | 94.7% |

**Paired comparison** (180 events tracked before by both): only SAM 3 kept the identity in 55,
only SAM 3.1 in 1 (McNemar exact p = 1.6e-15).

## Findings
1. **SAM 3 keeps identity through occlusion far better than SAM 3.1: 94% vs 63%.** The gap is
   occlusion-specific: without occlusion the two models are close (98% vs 95%).
2. **SAM 3.1's failure mode is re-identification, not re-detection.** It finds the person again
   (only 10 of 180 lost) but under a new ID (56 switches, vs SAM 3's 7). This matches its lower
   association recall in the benchmark (AssRe 66 vs 87).
3. **SAM 3.1 degrades quickly with occlusion length:** 80% under 0.5 s, about 50% for 0.5–2 s,
   36% at 2 s or more. SAM 3 stays at 93–99% up to 2 s, then drops to 73%.
4. Verified by hand on MOTS20-02, person 26: SAM 3 has ID 14 before and after, SAM 3.1 has ID 9
   then ID 83 (`results/example_MOTS20-02_person26.jpg`).

## Limits
- Pedestrians only, 4 videos, 245 events. The ≥ 2 s bin is small (n = 26 / 25; wide CIs).
- 24 of 228 MOTS20 people could not be linked to a MOT17 track (14 of them in MOTS20-11) and are
  excluded.
- 42 (SAM 3) and 65 (SAM 3.1) events were not tracked before the occlusion and are excluded from
  the rates; the paired comparison only uses events both models tracked.
- One configuration each, as benchmarked (SAM 3.1: `max_num_objects=128`). Not yet tested: whether
  the gap comes from SAM 3.1's multiplex memory under crowding (its log reports multiplex-bucket
  "subscription rate: 1350%" on MOTS20-02). That is a lead, not a finding.

Files: `results/events.csv` (every event, both outcomes), `results/occlusion_results.json` (all
numbers above), `occlusion_analysis.py`, `plot_occlusion.py`.
