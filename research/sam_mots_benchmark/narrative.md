## Key findings

- **SAM 3 associates better than SAM 3.1 on both benchmarks.** AssA: KITTI-MOTS car 86.0 vs 77.3, pedestrian 74.4 vs 66.7; MOTS20 pedestrian 75.9 vs 61.2. IDF1 and identity switches follow (MOTS20: IDF1 84.8 vs 68.3, IDSW 66 vs 152).
- **SAM 3.1 loses identities in different ways on the two benchmarks.** On MOTS20 (crowded, occluded pedestrians) the drop is in association *recall* (AssRe 86.6 vs 66.3): one person is split into several tracks. It predicts 300 identities for 228 real pedestrians, and 62 pedestrians are mostly lost (SAM 3: 18). Association *precision* is similar (85.3 vs 82.3). On KITTI-MOTS the loss is mainly extra tracks and false positives (cars: 1,233 FP vs 511).
- **The gap is not uniform.** Over the 19 sequence/class pairs, SAM 3 has an AssA more than 3 points higher on 10 and SAM 3.1 on 3 (KITTI 0007 and 0014 pedestrian, MOTS20-09). The other 6 are within 3 points. The largest SAM 3.1 losses are KITTI 0002 pedestrians (23 vs 59), MOTS20-11 (57 vs 84) and MOTS20-05 (52 vs 70).
- **Both zero-shot models beat the trained TrackR-CNN reference on association**, most of all for pedestrians (MOTS20 AssA 75.9 / 61.2 vs 37.4).
- **SAM 3.1 failure case:** on MOTS20-11 (frames 468-485) one SAM 3.1 track's mask breaks into up to ~3,000 fragments covering 20-45% of the frame. It is scored as-is.

**Implication for occlusion:** SAM 3.1's MOTS20 loss is objects being split into new tracks, which is what happens when identity is not recovered after an occlusion. The next step measures that directly, using MOT17's per-frame visibility labels.
