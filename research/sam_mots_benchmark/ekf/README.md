# An EKF identity layer on top of SAM 3 and SAM 3.1

SAM finds and segments every object in each frame and gives it an ID. This layer adds a small motion filter per
object. The filter predicts where the object will be next. If the object disappears and SAM later reports a **new** ID
at the place the filter predicted, the layer gives the **old** ID back.

It uses past frames only, so it can run live. It never changes a mask, only IDs.

Method fixed in advance: `PROTOCOL.md`. Code: `ekf_relink.py` (filter and layer, with a self-check that runs first),
`dev_fragments.py` (development data), `ekf_video.py` and `ekf_event_video.py` (videos).

## Result (11 Oct 2026)

Scored exactly like the main benchmark (TrackEval, official ground truth). HOTA is the overall score. AssRe falls when
one object is split over several IDs. AssPr falls when several objects share one ID. IDSW counts ID switches.

| MOTS20 train (pedestrians, crowds) | HOTA | AssRe | AssPr | IDF1 | IDSW | IDs used (228 people) |
|---|---|---|---|---|---|---|
| SAM 3 | **72.6** | 86.6 | 82.3 | 84.8 | 66 | 230 |
| SAM 3 + EKF | 71.2 | 86.6 | 78.8 | 82.6 | 65 | 216 |
| SAM 3.1 | 61.5 | 66.3 | 85.3 | 68.3 | 152 | 300 |
| SAM 3.1 + EKF | **62.9** | 71.2 | 81.7 | 70.6 | 109 | 238 |
| EKF alone on SAM 3 masks (SAM's IDs erased) | 44.7 | 36.2 | 60.5 | 45.5 | 1,735 | 727 |
| EKF alone on SAM 3.1 masks | 44.7 | 38.2 | 65.1 | 45.9 | 1,191 | 779 |

| KITTI-MOTS val, HOTA / IDSW | cars | pedestrians |
|---|---|---|
| SAM 3 | 82.6 / 4 | 69.8 / 5 |
| SAM 3 + EKF | 82.4 / 4 | 69.8 / 4 |
| SAM 3.1 | 74.6 / 22 | 63.3 / 6 |
| SAM 3.1 + EKF | 74.6 / 17 | 64.1 / 3 |
| SAM 3.1 + EKF, image-plane model | 74.4 / 17 | 64.6 / 2 |
| EKF alone on SAM 3 masks | 70.5 / 634 | 63.2 / 179 |

The KITTI rows are run 3: the filter now also measures how tall each mask looks (see "How far off is the filter").
Run 2, without it: SAM 3.1 + EKF 74.7 / 16 and 64.2 / 3, EKF alone 71.3 / 539 and 62.0 / 140; SAM 3 + EKF did not change.
Identity scores moved by 0.1 HOTA or less. MOTS20 uses the image-plane model, whose output is byte-identical.

### How far off is the filter, in metres
KITTI has 3-D labels made with a lidar, which the benchmark does not use. They were used here to check the filter's
positions, never to feed it. "1 s prediction" is where the filter said an object would be one second later, relative
to our own car, against where the label then put it.

| cars within 60 m, median (90th percentile) | position error | 1 s prediction error |
|---|---|---|
| development sequences, road contact only (run 2) | 3.4 m (16.8) | 1.7 m (6.6) |
| development sequences, with mask height (run 3) | 2.2 m (7.0) | 1.1 m (3.8) |
| val, SAM 3 masks, run 3 | 1.9 m (7.2) | 1.2 m (5.4) |
| val, SAM 3.1 masks, run 3 | 2.3 m (11.5) | 1.5 m (12.5) |

Why the change: on the development sequences, distance from where a car meets the road is typically 11 % off within
30 m but 25 % off beyond 45 m, because roads are not flat (the labelled underside of a car is 1.0 to 2.0 m below the
camera, not always 1.65 m). Distance from how tall the mask looks, assuming a 1.5 m car, is 7 % off beyond 45 m.
The filter now takes both.

What is still wrong: the position error includes about 2 m that is not an error (the label marks the middle of the
car, the mask meets the road at its near end). A van or lorry is taller than 1.5 m, so it is placed too near (in
sequence 0008 a 2.7 m van at 50 m is placed at 27 m), and its speed is then wrong too. Cars coming towards us fast
are predicted worst (median 3 to 6 m after one second). SAM 3.1's broken masks give the long tail.

Identity through occlusion on MOTS20 (`../occlusion/OCCLUSION_PROTOCOL.md`, unchanged; share of occlusion events after
which the person has the same ID):

| | SAM 3 | SAM 3 + EKF | SAM 3.1 | SAM 3.1 + EKF |
|---|---|---|---|---|
| all events (95 % interval) | 93.6 % (89-97) | 92.6 % (88-96) | 63.3 % (55-71) | 70.0 % (61-78) |
| hidden under 0.5 s | 99.0 % | 98.0 % | 80.2 % | 81.4 % |
| hidden 0.5-1 s | 93.3 % | 93.3 % | 53.5 % | 72.1 % |
| hidden 1-2 s | 93.8 % | 90.6 % | 50.0 % | 57.7 % |
| hidden 2 s or more | 73.1 % | 73.1 % | 36.0 % | 40.0 % |
| events repaired / broken by the layer | 0 / 2 | | 14 / 2 (McNemar exact p = 0.004) | |

### What this says
1. **The filter helps the weaker model.** With SAM 3.1 it repaired 14 of the 56 identities lost at an occlusion and
   broke 2, removed 43 of 152 ID switches, and raised HOTA by 1.4.
2. **It does not help SAM 3, and costs it 1.4 HOTA in crowds.** SAM 3 already keeps 94 % of identities through
   occlusion, so there is little to repair, and every wrong hand-back merges two people.
3. **The cost is always the same: wrong hand-backs.** AssPr fell by about 3.5 points for both models on MOTS20. Position alone
   cannot tell a returning person from a newcomer who appears in the same place.
4. **A motion filter cannot replace SAM's identity.** With SAM's IDs erased, the filter alone scores 44.7 HOTA on MOTS20
   against 72.6 and 61.5.
5. **Motion is only trustworthy for about a second.** On the development data, right hand-backs had a median gap of
   0.2-0.6 s and wrong ones 1.1-1.7 s. The gain is in hides of 0.5-1 s; hides of 2 s or more are untouched.
6. **It is cheap.** About 1 ms per frame on KITTI and 9-15 ms per frame on MOTS20 (mostly 1080p) on a laptop CPU, almost all of it
   unpacking masks. The filter maths is under 0.5 ms.

### Limits
- Settings were chosen on clean ground-truth masks from driving scenes. There is no unscored data with SAM's own errors,
  and none from crowds.
- Two runs were made on the scored data (see the amendment in `PROTOCOL.md`).
- The occlusion table does not punish wrong hand-backs. Read it together with AssPr.
- Road-plane model: flat road, camera pitch ignored, the inertial unit is treated as if it sat at the camera.
- MOTS20: two of the four videos have a moving camera and there is no calibration, so camera motion is not removed.
- One configuration, no confidence interval on HOTA.

## Every value in the filter

The filter maths (predict, update, Mahalanobis distance) comes from filterpy 1.4.5 (`ExtendedKalmanFilter`, the library
of R. Labbe's *Kalman and Bayesian Filters in Python*). `ekf_relink.py` only defines the model passed to it: the
state, F, Q, h(x), its Jacobian, R and our own car's motion. Swapping the hand-written maths for filterpy on 14 Oct
reproduced all nine saved outputs byte for byte (KITTI val, MOTS20 train and the development set).

There is one filter per object. `dt` is the time between frames: 0.1 s on KITTI, 1/30 s or 1/14 s on MOTS20.

### What the filter believes: the state `x` and its uncertainty `P`
| symbol | meaning | unit |
|---|---|---|
| `X` | how far the object is to the right of our camera | m |
| `Z` | how far ahead it is | m |
| `vX`, `vZ` | its speed over the ground, sideways and ahead | m/s |
| `P` (4 x 4) | how unsure the filter is about each of the four, and how their errors go together | |

On MOTS20 there is no calibration, so the same four numbers are in the image: `u`, `v` (pixels) and their speeds
(pixels per second).

### Prediction: how the belief moves forward one frame
| symbol | value | meaning |
|---|---|---|
| `F` | position + speed x `dt`, speed unchanged | "constant velocity": the object carries on as it was going |
| `Q` | `SIG_ACC`² x [[dt⁴/4, dt³/2], [dt³/2, dt²]] for each axis | process noise: how wrong "carries on as it was" can be, because real objects speed up, brake and turn |
| `SIG_ACC` | cars 3 m/s², people 1 m/s² | typical acceleration. Harder braking than this happens, and the filter then lags for a few frames |
| own-car motion | heading change, speed ahead, speed left, from KITTI's inertial unit (`oxts`) | after each step the state is turned and shifted into where our car now is, so a parked car has speed 0 |

### Measurement: what the camera gives us
| symbol | value | meaning |
|---|---|---|
| `z` | KITTI: `(u, v, s)` = centre column of the mask's box, row where it meets the road, its height in pixels. MOTS20: `(u, v)` = centre of the box | on KITTI two of the three say how far the object is: where it touches the road, and how big it looks |
| the box | taken from the mask's largest connected part | SAM masks carry stray specks that drag a plain box off the object |
| `h(x)` | `u = fx X / Z + cx`, `v = fy Hc / Z + cy`, `s = fy Ho / Z` | the camera equation: where a point on the road at (X, Z) lands in the image. It divides by Z, so it is not a straight-line relation. That is why this is an *extended* Kalman filter |
| `fx, fy, cx, cy` | from each sequence's KITTI calibration file (0008: 721.5, 721.5, 609.6, 172.9) | focal length and image centre, in pixels |
| `Hc` (`CAM_HEIGHT`) | 1.65 m | camera height above the road |
| `Ho` (`HEIGHT`) | cars 1.5 m, people 1.7 m | the object's own height; a real van is taller, which is this cue's weakness |
| `H` | [[fx/Z, -fx X/Z², 0, 0], [0, -fy Hc/Z², 0, 0], [0, -fy Ho/Z², 0, 0]] | the slope of `h` at the current belief (its Jacobian); the self-check compares it with a numerical derivative |
| `R` | (2 px + 10 % of box width)², (2 px + 10 % of box height)², and on KITTI (1 px + 8 % of box height)² for `s` | measurement noise: masks wobble, and big masks wobble by more pixels. The 8 % covers real cars not all being 1.5 m tall; `SIZE0`, `SIZEREL` were set on the development sequences |
| `CUT`, `SHRUNK` | 5x noise when the box touches the image edge or is under 80 % of its remembered height | a cut or half-hidden box no longer shows the true centre or road contact |
| remembered height | the largest recent box height, forgetting 20 % per second | tells "half hidden" from "far away" |

On MOTS20 `h(x)` is simply (u, v), a straight-line relation, so there the same code is an ordinary Kalman filter.

### Update: mixing prediction and measurement
| symbol | formula | meaning |
|---|---|---|
| `y` | `z - h(x)` | the surprise: measured minus predicted |
| `S` | `H P Hᵀ + R` | how big a surprise the filter expects |
| `K` | `P Hᵀ S⁻¹` | the gain: how far to move towards the measurement. Unsure belief or precise measurement gives a big move |
| new `x`, `P` | `x + K y`, `(I - K H) P (I - K H)ᵀ + K R Kᵀ` | the corrected belief, now more certain |

### A new object
Position: the camera equation solved backwards from its first mask. On KITTI the two distance cues are averaged,
each weighted by how sure it is. Position uncertainty: `R` carried through that same equation, so far objects start with a long ellipse along the viewing direction. Speed: 0 ± `SIG_V0` (cars 10 m/s,
people 1.5 m/s), because one picture says nothing about speed.

### Giving an ID back
| symbol | value | meaning |
|---|---|---|
| `d²` | `yᵀ S⁻¹ y` | distance between a new detection and a lost track's prediction, counted in the filter's own uncertainty |
| `GATE` | 11.83 (MOTS20), 14.16 (KITTI) | if the track's belief is right, 99.73 % of its true detections have `d²` below this (chi-square for 2 or 3 measured numbers) |
| choice among several | smallest `d² + ln det S` | closest first; between equals, the track that is surer |
| `MEMORY` | 1 s | a lost track is forgotten after this |
| in the picture | the prediction must fall inside the image | an object expected outside the picture cannot be the one that just appeared |
| `HEIGHT` | cars 1.5 m, people 1.7 m | MOTS20 only: turns `SIG_ACC` and `SIG_V0` from metres into pixels, using the object's own height in the image as the ruler |

An ID that SAM itself keeps is never changed.

## Run it
```bash
# data: TrackEval data.zip, MOT17Labels.zip, KITTI data_tracking_calib.zip and data_tracking_oxts.zip, each unpacked
# into _data/x_<name>/ ; SAM outputs copied to _data/trk/ (see PROTOCOL.md for checksums)
U="uv run --python 3.10 --with numpy==1.23.5 --with scipy --with pycocotools --with filterpy"
$U python ekf_relink.py                                               # self-check only
$U python ekf_relink.py kitti _data/trk/kitti_mots/SAM3.1 _data/trk/kitti_mots/SAM3.1_EKF
$U python ekf_relink.py mots  _data/trk/mots/MOTS-train/SAM3.1 _data/trk/mots/MOTS-train/SAM3.1_EKF
$U --with opencv-python-headless --with "imageio[ffmpeg]" --with pillow python ekf_video.py _data/trk/kitti_mots/SAM3.1 0010 <image_02 folder> out.mp4
```
Then score with TrackEval as in `../evaluate.sh`.

## Next
A check on *who* it is, not only *where*: appearance from SAM's own features under the mask, tested first on sequences
that are not scored (SAM run on the KITTI-MOTS training split and on MOT17 videos outside MOTS20).
