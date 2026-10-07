# Identity through occlusion: SAM 3 vs SAM 3.1 on MOTS20 (train)

Fixed before any result of this analysis was computed (30 Sep 2026). Question: **when a
pedestrian is occluded and reappears, does the tracker give them the same identity?**
Inputs are the exact tracker files scored by TrackEval in the main benchmark (see
`../PROTOCOL.md`); nothing is re-run.

## Data
- Masks and identities: MOTS20 train ground truth (TrackEval `data.zip`), class 2
  (pedestrian); class 10 (ignore) is excluded.
- Occlusion: MOT17 train ground truth for the same four videos (MOTS20-02/05/09/11 =
  MOT17-02/05/09/11), `MOT17Labels.zip` from motchallenge.net (sha256
  `0aa79322e91583369f42f17c4d79a0b145380d8732487bba59272048dc82b2b9`; the gt is identical in the
  DPM/FRCNN/SDP copies). Rows with class 1 (pedestrian) and consider-flag 1. Each box carries a
  visibility ratio in [0, 1]; boxes are annotated through full occlusion. Frame rates from
  `seqinfo.ini` (30, 14, 30, 30 fps).

## Linking MOTS20 people to MOT17 tracks
MOTS20 masks were drawn for MOT17 pedestrians, but track IDs are not guaranteed to agree.
In each frame, MOTS20 masks are matched one-to-one (Hungarian) to MOT17 boxes with visibility
>= 0.5 by IoU of the mask's bounding box with the box, keeping pairs with IoU >= 0.5. Each
MOTS20 person gets the MOT17 track it matched most often (majority vote). Reported: the share of
matched person-frames that agree with the majority, and how many people could not be linked.

## Occlusion events (from MOT17 visibility only)
For a linked person, an **event** is a maximal run of consecutive frames with visibility < 0.25,
with the frames immediately before and after the run both visible (>= 0.25) and in the track.
- **duration**: run length in seconds (frames / fps). Bins: < 0.5 s, 0.5-1 s, 1-2 s, >= 2 s.
- **depth**: the minimum visibility in the run. Partial: >= 0.1; heavy/full: < 0.1.

## Tracker identity at a frame
Per frame, ground-truth masks are matched one-to-one (Hungarian) to the tracker's masks by
mask IoU, keeping pairs with IoU >= 0.5 (the CLEAR-MOT matching criterion).

## Outcome of an event, per tracker
- **pre-ID**: the tracker ID matched to the person in the last matched frame within 1 s before
  the event. If there is none, the event is *not tracked before* and is excluded from the rates
  (counted separately).
- **post-ID**: the tracker ID in the first matched frame within 2 s after the event.
- **kept** = post-ID equals pre-ID; **switched** = a different ID; **lost** = no match within 2 s.

## Metrics
- **ID recovery rate** = kept / (kept + switched + lost). Primary metric, overall and by
  duration and depth.
- **Re-association rate** = kept / (kept + switched): given the person is found again, is it
  under the same ID?
- 95% confidence intervals: percentile bootstrap, resampling *people* (all of a person's events
  together), 10,000 resamples, fixed seed.
- **Paired comparison** on events tracked before by both models: counts where only SAM 3 kept
  the ID and where only SAM 3.1 did (McNemar exact test).
- **Control (no occlusion)**: non-overlapping 1-s windows in which the person is matched at the
  start and stays visible (>= 0.5) throughout; outcome at the end of the window is kept / switched
  / lost as above, and the same rate is reported. This separates identity loss caused by
  occlusion from identity drift in general.

All thresholds above are fixed here and are not tuned on the results.

## Amendment (30 Sep 2026, after the first results)
TrackR-CNN (the official MOTS20 reference outputs in TrackEval's `data.zip`) is added as a third
tracker so every table has the benchmark reference. The bootstrap now uses an independent random
stream per (subset, tracker), so a confidence interval does not depend on which trackers are
analysed. All definitions and thresholds above are unchanged.
