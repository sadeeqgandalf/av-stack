# Test-set runs: SAM 3 and SAM 3.1 as shipped (12 Oct 2026)

## What is run
| | |
|---|---|
| models | SAM 3 (`build_sam3_video_predictor()`), SAM 3.1 (`build_sam3_multiplex_video_predictor(max_num_objects=128, use_fa3=False)`), both with the settings they ship with |
| data | KITTI MOTS test (29 sequences, 11,095 frames) and MOTS20 test (MOTS20-01, 06, 07, 12; 3,044 frames) |
| prompts | "car", "pedestrian" (KITTI); "pedestrian" (MOTS20), as in `../PROTOCOL.md` |
| runner | `../run_sam_mots.py --benchmark kitti_mots_test mots_challenge_test`, one worker per model on one A100 80 GB |
| code | `ComputerVision_Research` commit 1fc7b28, torch 2.14.1+cu130 |
| outputs | `_data/test_runs/trackers_test/<benchmark>/<tracker>/data/<sequence>.txt`, copied from the pod every five minutes |

Test labels are withheld. These files can only be scored by the benchmark servers. **Nothing has been uploaded.**
Every score in `README.md`, `SETTINGS_EXPERIMENT.md` and `../narrative.md` is on a val or train split.

## SAM 3.1 cannot finish some sequences
| sequence | frame where it stops |
|---|---|
| KITTI tracking training 0019 (practice split), twice | 226 |
| KITTI-MOTS val 0010, with SAM 3's settings | 254 |
| KITTI tracking test 0010 | 667 |
| KITTI tracking test 0014 | 55 |
| MOTS20-07 (test) | 272 |
| MOTS20-12 (test) | 147 |

**What happens.** SAM 3.1 keeps tracked objects in groups that share one memory. In long or busy sequences a group
can be left with objects but none of the frames it was started from ("conditioning frames"). Nothing downstream is
written for that case.

**Three layers, one cause.**
1. *Meta's code.* It raises `No points are provided` in the tracker.
2. *The copy in `ComputerVision_Research` since commit 1fc7b28 (7 Oct).* A branch in
   `sam3/sam3/model/sam3_multiplex_base.py` (line 1752) catches that state, logs
   `empty cond_frame_outputs with N obj_ids; emitting inert masks`, and skips the group for the frame. The group then
   has no output for the frame, and the next step raises `conditioning objects not found`
   (`video_tracking_multiplex_demo.py`, `_run_memory_encoder`). This is the error seen in every run of 11-12 Oct.
3. *A fix tried on 12 Oct and removed.* Replacing that `raise` with an empty set let the step pass, and the next one
   (`add_new_masks` -> `_run_single_frame_inference`) raised `No existing output found for frame N`. All four
   sequences it was tried on stopped at the same frame as before.

So neither change repairs it: each moves the failure one step later. A real repair has to give the group a valid
output for that frame, or re-seed it, inside SAM 3.1's group bookkeeping. That has not been attempted.

**Does the 7 Oct branch affect the scores already reported?** No. Its warning line appears in the logs of the six
runs above and in no run that completed (checked in every SAM 3.1 log of 11-12 Oct). A sequence that completed never
entered the branch, so its output is what Meta's code produces. Two val sequences rerun on 11 Oct were byte-identical
to the 30 Sep outputs.

**What this means for SAM 3.1 on the test sets.** As shipped it completes 27 of 29 KITTI test sequences and 1 of 4
MOTS20 test sequences (MOTS20-01). MOTS20-07 and -12 stop as above; MOTS20-06 runs out of memory (next section).
There is no complete SAM 3.1 submission for either benchmark.

## GPU memory: two sequences do not fit in 80 GB
Memory grows with every frame, faster in busy scenes, and the growth does not stop:

| sequence | model | stopped at | GPU memory then |
|---|---|---|---|
| MOTS20-06 (1,194 frames, 640 x 480) | SAM 3 | frame 912 | 75 GB |
| MOTS20-06 | SAM 3.1 | frame 896 | 74 GB |
| KITTI test 0014 (850 frames) | SAM 3 | frame 742 | 75 GB, while sharing the card |

At that rate MOTS20-06 needs about 100 GB as one sequence. It is the longest sequence either model has been given;
the longest that completed has 1,176 frames (KITTI test 0010, a quieter scene, 48 GB). Both are **held**: not run
again until the growth is reduced or a larger GPU is used. Without MOTS20-06 there is no complete MOTS20 test
submission for either model. SAM3-ASH (arXiv 2610.01022) reports the same limit and avoids it by cutting videos
into 50-frame chunks.

What already runs off the GPU: the video frames, SAM 3's inference state, SAM 3.1's per-frame tracker outputs, and
the cached full-resolution masks of frames already written (`../run_sam_mots.py`, `SESSION_OPTIONS`,
`offload_frame_cache`). What still grows has not been identified.

## What went wrong in the scheduling (about 4 hours lost)
- The memory estimates fitted on val were too low for test, so the two models overfilled the card together
  (2 failures).
- After the estimates were raised, long sequences could not overlap, SAM 3 always took the next slot, and SAM 3.1
  waited 80 minutes.
- Both workers then took turns failing on MOTS20-06 for 3 hours (7 attempts), because a worker retries a sequence
  that ran out of memory and nothing limited the retries.

- A restart at 19:09 would have run "as shipped" SAM 3.1 with the 12 Oct fix loaded; it was stopped after 2 minutes,
  before any file was written.

From 19:11 UTC the queue (`queue2.sh` on the pod) runs one model at a time and holds a sequence after its first
memory failure. A sequence stopped for memory is rerun from its first frame; no partial output is kept.

## Results (runs ended 21:00 UTC, 12 Oct)
| | KITTI MOTS test (29) | MOTS20 test (4) |
|---|---|---|
| SAM 3 as shipped | 28 completed; 0014 held (memory) | 3 completed (01, 07, 12); 06 held (memory) |
| SAM 3.1 as shipped | 27 completed; 0010, 0014 stop (group fault) | 1 completed (01); 07, 12 stop (group fault); 06 held (memory) |

- Output files: `_data/test_runs/trackers_test/`. Logs: `_data/test_runs/logs/`.
- Speed alone on the A100: SAM 3 0.4 to 1.4 s per frame; SAM 3.1 0.4 to 0.8 s per frame.
- Neither model has a complete set for either benchmark. For SAM 3 the gap is two sequences that need more than
  80 GB. For SAM 3.1 it is also four sequences it cannot finish.
- Nothing has been uploaded or scored.
