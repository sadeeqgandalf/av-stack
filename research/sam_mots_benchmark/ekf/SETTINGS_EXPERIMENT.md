# Is SAM 3.1 behind SAM 3 because of its settings? (11-12 Oct 2026)

## Question
SAM 3 scored higher than SAM 3.1 on both benchmarks (`../narrative.md`). Both were run with the settings they ship
with, and those settings differ (`sam3/model_builder.py`: SAM 3 lines 756-772 and 494, SAM 3.1 lines 1068 and
1192-1214). How much of the gap do the settings explain?

## What differs
| setting | SAM 3 | SAM 3.1 | what it does |
|---|---|---|---|
| `score_threshold_detection` / `new_det_thresh` | 0.5 / 0.7 | 0.4 / 0.65 | how sure a detection must be to count, and to start a new track |
| `init` / `max` / `min_trk_keep_alive` | 30 / 30 / -1 | 0 / 8 / -4 | credit a track has before it is switched off for not matching a detection |
| `suppress_unmatched_only_within_hotstart` | on | off | on: that switching-off only happens in a track's first frames |
| `suppress_det_close_to_boundary` | off | on | drop detections at the image edge |
| `fill_hole_area` | 16 | 0 | clean small holes and specks in masks |
| `masklet_confirmation_enable` | off | on | wait for 3 detections in a row before a track is shown |
| `use_memory_selection` (tracker) | on | off | keep memory frames in which the object was clearly visible |

## Variants (`../run_sam_mots.py --variant`)
The official build is unchanged; attributes are set on the built model and logged in `run_info_*.json`.
- **S**: SAM 3.1 with the first six rows set to SAM 3's values. Nothing was tuned: the values are the ones SAM 3 ships with.
- **M**, **SM**: memory selection on. **Did not run.** The path is unfinished in the shipped SAM 3.1 code: it raises
  `KeyError: 'iou_score'` on the first frame, and after a four-line fix for that (in
  `video_tracking_multiplex_demo.py`; not kept) `KeyError: 'multistep_pred_ious'`. Not pursued further.

## Data and order
Practice first: KITTI-MOTS `training_minus_val` (12 sequences, not scored in the benchmark). Then S once on
KITTI-MOTS val. No crowded-pedestrian practice data exists with mask ground truth, so MOTS20 is not covered.
A100 80 GB, torch 2.14.1+cu130, sam3 code at `ComputerVision_Research` commit 1fc7b28, one worker per model.

## Practice result (11 sequences)
SAM 3.1 as shipped crashed on the twelfth, 0019 (`ValueError: conditioning objects not found at frame_idx=226`,
inside SAM 3.1's own code); S and SAM 3 completed it. The table uses the 11 sequences all three completed
(seqmap `dev11`).

| cars | HOTA | DetA | DetRe | DetPr | AssA | IDSW | false positives | misses |
|---|---|---|---|---|---|---|---|---|
| SAM 3 | 77.2 | 76.2 | 85.7 | 83.0 | 78.4 | 58 | 1,656 | 1,060 |
| SAM 3.1 | 73.5 | 69.4 | 79.9 | 80.0 | 78.1 | 85 | 2,116 | 2,140 |
| SAM 3.1, variant S | 75.0 | 72.6 | 80.5 | 83.8 | 77.8 | 82 | 1,384 | 2,090 |

| pedestrians (37 people) | HOTA | DetA | AssA |
|---|---|---|---|
| SAM 3 | 65.7 | 60.5 | 72.3 |
| SAM 3.1 | 66.3 | 60.8 | 72.9 |
| SAM 3.1, variant S | 64.3 | 61.4 | 68.1 |

- On these sequences the gap is in detection, not association (AssA 78.4 vs 78.1).
- SAM 3's settings close 1.5 of the 3.7 HOTA points for cars. They remove SAM 3.1's extra false positives
  (2,116 -> 1,384, below SAM 3) but not its misses (2,140 -> 2,090, twice SAM 3's).
- Car HOTA rose on 8 of the 10 sequences with cars, was equal on one and fell on one (0011).
- Pedestrians got worse, on 37 people: too few to conclude from.
- Cost: variant S ran at 1.0-1.3 s per frame against 0.36, and peaked at 62 GB of GPU memory.

## Scored result: KITTI-MOTS val, variant S run once (8 of 9 sequences)
Variant S crashed on 0010 with the same error as above (`conditioning objects not found at frame_idx=254`), so all
three are compared on the other 8 sequences (seqmap `val8`). SAM 3 and SAM 3.1 are the outputs of 30 Sep. Check that
tonight's environment matches that run: shipped SAM 3.1 rerun on 0013 and 0018 gave byte-identical files.

| cars | HOTA | DetA | AssA | IDF1 | IDSW | false positives | misses |
|---|---|---|---|---|---|---|---|
| SAM 3 | 82.3 | 79.3 | 85.6 | 92.8 | 4 | 478 | 306 |
| SAM 3.1 | 74.0 | 72.0 | 76.4 | 83.7 | 22 | 1,157 | 436 |
| SAM 3.1, variant S | 78.3 | 76.0 | 81.0 | 87.9 | 14 | 755 | 396 |

| pedestrians | HOTA | DetA | AssA | IDF1 | IDSW | false positives | misses |
|---|---|---|---|---|---|---|---|
| SAM 3 | 70.2 | 66.5 | 74.5 | 90.1 | 5 | 285 | 348 |
| SAM 3.1 | 63.6 | 60.9 | 66.8 | 81.8 | 6 | 444 | 488 |
| SAM 3.1, variant S | 68.1 | 65.5 | 71.2 | 87.8 | 5 | 243 | 450 |

- SAM 3's settings recover 4.3 of the 8.3 HOTA points for cars and 4.5 of the 6.6 for pedestrians.
- Here association moves too (cars 76.4 -> 81.0, pedestrians 66.8 -> 71.2), unlike on the practice sequences.
- The gain is uneven: cars 0013 56.3 -> 80.3 and 0018 65.7 -> 83.3 reach SAM 3's level; 0008 and 0014 do not improve.
- Speed on val: 0.25-0.73 s per frame, peak 22-42 GB (lighter than on the practice sequences).

## What this says
About half of SAM 3.1's deficit on KITTI-MOTS comes from the tracking settings it ships with, not from its network.
The rest remains with SAM 3's settings. The memory-selection difference could not be tested, because that path does
not run in the shipped SAM 3.1 code. Two sequences (practice 0019 as shipped, val 0010 with variant S) hit a crash
inside SAM 3.1's memory encoder.

## Not done
MOTS20 (crowded pedestrians, 1080p): no practice data with mask ground truth, and variant S peaked at 62 GB on
KITTI-sized images, so 1080p may not fit an 80 GB GPU. The EKF layer on top of variant S.
