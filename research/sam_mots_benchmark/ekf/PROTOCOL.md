# EKF identity layer on SAM 3 / SAM 3.1: protocol

Fixed on 11 Oct 2026, before the layer was run on any scored sequence.

## Question
SAM 3.1 finds a person again after an occlusion but often under a new ID (56 of 180 events on MOTS20; SAM 3: 7).
Does a motion filter that predicts where each lost object will be, and hands the old ID back when a new ID shows up
there, repair this? And does it do harm where SAM was already right?

## What is compared
| name | identity decided by |
|---|---|
| SAM3, SAM3.1 | the model alone (the files scored in `../PROTOCOL.md`, unchanged) |
| SAM3_EKF, SAM3.1_EKF | the model, plus the layer: a new SAM ID inside a lost track's gate gets the lost track's ID |
| SAM3_EKFalone, SAM3.1_EKFalone | the filter alone: SAM's masks with its IDs erased; position is the only cue |
| SAM3_EKFpixel, SAM3.1_EKFpixel (KITTI only) | as _EKF, with the image-plane model in place of the road-plane model |

Masks are never changed, so detection scores (DetA) cannot move. Only IDs change.

## Scored data and metrics
The same as the main benchmark: KITTI-MOTS val (9 sequences) and MOTS20 train (4 videos), official ground truth,
TrackEval at the same commit. HOTA, AssA, AssRe, AssPr, IDF1, ID switches. On MOTS20 also the identity-through-occlusion
analysis of `../occlusion/OCCLUSION_PROTOCOL.md` (245 events), unchanged.

Evaluator difference: this Mac runs TrackEval with Python 3.10 / numpy 1.23.5 (the pinned numpy 1.18.1 has no Apple
Silicon build). Check: it reproduces the published SAM 3 and SAM 3.1 summary files to every printed digit.

## Models
- KITTI-MOTS: road-plane EKF. State in metres, nonlinear camera measurement, own-car motion from the KITTI
  inertial unit (`oxts`), calibration from the KITTI `calib` files.
- MOTS20: no calibration exists, so the same filter runs in the image plane (pixels), where the measurement is linear.

## Settings and how they were chosen
Every value is in the header of `ekf_relink.py`. Noise values were set from physics before any run. Three were then
chosen on development data that is not scored: the ground truth of the 12 KITTI-MOTS training sequences outside the
val split, with a new ID forced after every gap (`dev_fragments.py`; 98 gaps).

| setting | tried | chosen | why |
|---|---|---|---|
| MEMORY | 1, 2, 3 s | 1 s | wrong hand-backs had a median gap of 1.1-1.7 s, right ones 0.2-0.6 s |
| CUT (noise for cut or shrunken boxes) | off, 5x | 5x | boxes cut by the image edge or partly hidden were 46 % of the filter's worst predictions |
| likelihood threshold on a hand-back | 5 values | removed | no gain once MEMORY was 1 s |

Also tried on the development data and left out: a colour-histogram check (little gain over motion alone).

Development result with the chosen settings (cars / pedestrians): HOTA 97.1 -> 98.0 / 91.4 -> 93.6, ID switches
56 -> 32 / 42 -> 15; hand-backs right 24 / 27, wrong 8 / 6, missed 29 / 12.

Limit of this choice, known in advance: there is no unscored data with SAM's own errors, and none from MOTS20-like
crowds. The settings come from perfect masks on driving scenes.

## Run once
Each scored configuration is run once with these settings. Anything changed afterwards is reported as a change.

## Amendment (11 Oct 2026, after the first run)
The first run measured each object from the plain bounding box of its mask. Looking at the first video showed that
SAM masks often carry stray specks far from the object, which put the box in the wrong place: 7 % (KITTI) and 12 %
(MOTS20) of SAM 3 masks, 21 % and 38 % of SAM 3.1 masks. The development data has clean masks, so it could not show
this. Change: the box is taken from the mask's largest connected part (`main_box`). No setting was changed. This was
found from mask geometry, not from the scores, but it is a second run on scored data and both runs are reported.

| MOTS20 train, HOTA / AssRe / AssPr / IDSW | run 1 (plain box) | run 2 (largest part) |
|---|---|---|
| SAM3.1_EKF | 61.5 / 71.5 / 76.9 / 109 | 62.9 / 71.2 / 81.7 / 109 |
| SAM3_EKF | 70.4 / 86.3 / 77.2 / 67 | 71.2 / 86.6 / 78.8 / 65 |

KITTI-MOTS moved by 0.6 HOTA or less in either direction. All results in `README.md` are run 2.
Development result after the change: HOTA 98.0 / 92.8, ID switches 32 / 17, hand-backs right 24 / 25, wrong 8 / 7.

## Amendment 2 (11 Oct 2026): the mask's height as a third measurement, KITTI only
Reason: the first HUD video printed how far each one-second prediction missed, and oncoming cars missed by 14 to
20 m. That was seen on a scored sequence (0008). The cause and the fix were then worked out on the development
sequences only, against KITTI's 3-D labels (which the benchmark does not use): distance from the road-contact row
alone is 25 % off beyond 45 m, distance from the mask's height is 7 % off there.

Change: the road-plane model measures three numbers per mask instead of two (`SIZE0`, `SIZEREL` = 1 px, 8 %, set
from the spread of labelled car heights on the development sequences; 2 px, 3 px, 5 % and 12 % were tried there and
were worse). The gate became the chi-square value for three numbers (14.16). The image-plane model is untouched:
its MOTS20 output is byte-identical, so no MOTS20 number changed.

Development result (cars / pedestrians): HOTA 98.0 -> 98.0 / 92.8 -> 92.9; hand-backs right 24 -> 23 / 25 -> 21,
wrong 8 -> 6 / 7 -> 3. Position error against the labels 3.4 -> 2.2 m, one-second prediction error 1.7 -> 1.1 m.

Third run on KITTI-MOTS val, once, HOTA / IDSW (cars, pedestrians):

| | run 2 | run 3 |
|---|---|---|
| SAM3_EKF | 82.4 / 4, 69.8 / 4 | 82.4 / 4, 69.8 / 4 |
| SAM3.1_EKF | 74.7 / 16, 64.2 / 3 | 74.6 / 17, 64.1 / 3 |
| SAM3_EKFalone | 71.3 / 539, 62.0 / 140 | 70.5 / 634, 63.2 / 179 |
| SAM3.1_EKFalone | 66.8 / 610, 58.0 / 133 | 65.9 / 649, 58.9 / 161 |

Run 2 outputs are kept as `_data/trk/kitti_mots/*_run2`. All KITTI results in `README.md` are run 3.
The 3-D labels of the val sequences were used to report position and prediction error, and to find out why one
object in 0008 was misplaced (it is a 2.7 m van). No setting was chosen on them.
