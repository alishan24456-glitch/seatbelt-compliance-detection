# Seatbelt Compliance Detection (traffic enforcement)

A roadside / traffic-camera seatbelt check: the same idea that sits
behind real enforcement cameras (Acusensus / Monocam systems are live
in Australia and Germany). YOLOv8n finds the vehicles, the windshield
band of each vehicle box is split into two seat regions (driver /
passenger side), and a diagonal-belt check across each occupant's
torso gives a **BELTED / NO BELT** verdict per seat, with a compliance
board and a timestamped event log.

## How a seat is judged

- **Windshield band** — the upper part of the vehicle box
  (x 0.10–0.90, y 0.06–0.42 of the box). It is a documented **image
  region for this scene**, not a calibrated camera model: on a
  front-facing car the windscreen sits in the upper part of the box.
- **Belt check** — inside each seat's torso region (the lower 65% of
  the seat) the pipeline runs Canny edges + Hough line segments and
  looks for a diagonal line (20–65° from horizontal, either
  direction) that spans at least 45% of the torso height. A 3-point
  belt runs shoulder-to-hip, so a genuine strap is one long diagonal;
  a window frame, a headrest edge or a reflection is short,
  horizontal or vertical. The check scores the total qualifying
  diagonal length against the torso diagonal: **BELTED at ≥ 0.45**,
  otherwise **NO BELT**.
- **Honesty gates** — a verdict is only attempted when the seat
  region is big enough to judge (at least 40×35 px) and neither too
  dark (mean < 60) nor blown out (mean > 245). A perfectly uniform
  torso region also reads **UNCLEAR**: it is an empty seat (bare
  glass), not an occupant. A vehicle box clipped by the frame edge
  reads UNCLEAR, because its windscreen is out of frame. There is
  deliberately no minimum-edge gate: a bright, clearly visible torso
  with no diagonal strap is a NO BELT verdict.
- **Confirmation** — a seat verdict must repeat in **3 consecutive
  detection rounds** before it counts and is logged, so one
  flickering YOLO box or one glinting reflection cannot raise a flag.
- **Tracking** — a centroid tracker with a scale-aware match gate:
  a car driving straight at the camera grows fast (its centroid
  jumps ~95 px between detection rounds on the demo clip), so the
  match distance is half the bigger box's size instead of a fixed
  80 px. A fixed gate spawned a new track for the same car every few
  rounds (32 "vehicles" for ~10 real cars on the first demo run).

A **NO BELT verdict is a flag FOR HUMAN REVIEW, never a fine.**
The banner and the event log both say so on every entry.

## States

| State | Meaning |
|---|---|
| `BELTED` | A diagonal strap was found across the torso (score ≥ 0.45) |
| `NO BELT` | No diagonal strap visible in the seat region — flagged for human review |
| `UNCLEAR` | The seat cannot be judged: too small, too dark, blown out, uniform (empty), or out of frame |
| `--` | No confirmed verdict yet (the seat is still being checked) |

The board shows vehicles tracked, seats assessed, BELTED / NO BELT /
UNCLEAR counts, and the compliance rate among decided seats
(belted / (belted + no belt)).

## Demo (real footage)

Real roadside footage (Pexels video 7390231, free license — title and
creator not recorded, the Pexels video page is blocked by a Cloudflare
check): an elevated front view of cars driving toward the camera on a
highway, two lanes, steady flow. 30 s (first 30 s of a 38.9 s clip),
900 frames, 300 tracker updates, 77 vehicle detections,
**14 vehicles tracked**.

- **Seats assessed: 15 — BELTED 0, NO BELT 6, UNCLEAR 9.**
- **Compliance among decided seats: 0.0% (0/6).**
- **NO BELT events: 6**, first at 0.4 s. All six are seat B (the
  passenger side); every seat A on this clip reads UNCLEAR — the
  driver side is systematically darker (steering wheel, driver
  silhouette and shadow put the seat-A regions at mean 39–65, below
  or just above the brightness gate), while the passenger side reads
  mean 61–98.
- Two seats scored just under the belt threshold (0.38 and 0.44
  against 0.45) and read NO BELT — borderline calls, flagged for
  review.

## Honest limits — read this before trusting the board

- **No strap is visible on any seat in this clip.** The 0% compliance
  figure means "0 of 6 judged seats showed a strap", not "6 unbelted
  drivers". At this distance, through dark glass, a belt is barely
  readable even to a human looking at the same pixels.
- **The pipeline cannot tell an empty seat from an unbelted
  occupant.** YOLOv8n detects no persons through the windscreens on
  this clip (0 person detections at every sampled timestamp), so
  there is no occupant-presence signal to gate on. Most of the six
  NO BELT flags are probably empty passenger seats, not unbelted
  passengers. Every flag is for human review for exactly this reason.
- A strap the same colour as the clothing, or hidden by a reflection
  on the glass, can read NO BELT. A strap at a shallow angle on a
  very wide torso can fall below the 20° diagonal gate.
- The windshield band is an image region, not a camera calibration:
  on a side-on car the windows sit lower in the box and the seats
  read UNCLEAR rather than a wrong verdict.
- The compliance rate is a property of this clip's 6 judged seats,
  not a measurement of real-world compliance.

## Simulation (logic check only — not the demo)

`--simulate` renders a roadside view with three cars driving toward
the camera: car 1 has a belted driver and a belted passenger, car 2
a beltless driver and a belted passenger, car 3 a beltless driver and
an empty passenger seat. The vehicle source finds the cars with a
colour mask + contours — the windshield band, the belt check and the
verdict logic are the exact same code as real mode, and the detector
never sees the simulation's ground truth.

Verified on the final build (30 s, 900 frames): **3 vehicles tracked,
6 seats assessed — BELTED 3, NO BELT 2, UNCLEAR 1 (the empty seat),
compliance 60.0% (3/5), 2 NO BELT events, first at 3.6 s.** The log
reads exactly as scripted: car 1 both seats BELTED at 0.8 s, car 2
driver NO BELT and passenger BELTED at 3.6 s, car 3 driver NO BELT at
16.9 s, and each car's seats go UNCLEAR as it slides out of frame.

## Bugs found and fixed during verification

1. **HoughLinesP shape (OpenCV 5):** `lines[:, 0]` yields scalars
   because OpenCV 5 returns `(N, 4)`, not `(N, 1, 4)` — the first
   simulation run crashed. Fixed by reshaping to `(-1, 4)`.
2. **CLAHE destroyed the belt edge:** CLAHE on a two-tone torso
   (skin 197 vs strap 245) flattened the step to ~18 grey levels and
   Canny found zero edges, so every seat read "no edges to judge".
   Removed; the check runs Canny on the raw ROI.
3. **Belt angle gate too tight (25°):** the torso region is ~2:1
   wide:tall, so a shoulder-to-hip strap crosses it at ~23° and every
   belted seat scored 0.00. Lowered to 20° (window frames and
   headrest edges are near 0°/90°, so they are still excluded).
4. **Sim cars merged into one blob:** every car slid in from the left,
   so car 2 drove straight through car 1's box and the colour mask
   saw one merged contour; the tracker spawned duplicate tracks.
   Cars now slide in from their nearest edge, and the timeline is
   staged so a leaving car never crosses another car's position
   (car 2 leaves at 14 s, car 3 enters at 16 s).
5. **Seat counting counted the first-ever verdict:** a seat that first
   read UNCLEAR while sliding into frame and was later decided
   counted as UNCLEAR, so the board read 1 BELTED / 1 NO BELT /
   8 UNCLEAR while the log showed three BELTED confirmations.
   Outcomes are now tracked per seat: a seat counts its first
   *decided* verdict, and a later decided verdict overwrites an
   earlier UNCLEAR.
6. **Fixed tracker gate broke on head-on cars:** the 80 px centroid
   gate (borrowed from the blind-spot tracker) could not follow a
   car driving at the camera (95 px jumps, box IoU 0.046 between
   rounds) — 32 "vehicles" for ~10 real cars. Replaced with a
   scale-aware gate (half the bigger box's size) and a 0.02 IoU floor.
7. **Empty seat read NO BELT:** the simulated empty passenger seat is
   uniform dark glass (mean ~41) and "no diagonal" became a false
   accusation. Added a uniformity gate (std < 6 → UNCLEAR) and gave
   the simulated torso non-diagonal texture (collar line, centre
   seam) so an occupied torso still reads as occupied.
8. **Brightness gate too low (30):** dark-glass seats (mean 39–58)
   were all judged NO BELT and the board read 0% compliance. Raised
   to 60; dark glass now reads UNCLEAR.
9. **Overlapping seat labels:** one tag per seat printed on top of
   the other into unreadable mush and clipped off the top of small
   cars. Seat verdicts are now one combined line in the vehicle tag,
   and the tag shows confirmed verdicts only (an earlier build showed
   the pending verdict, so a seat could read BELTED in the tag while
   the board counted 0 BELTED).

## Files

- `seatbelt_compliance_detection.py` — the detector
  (`--source pick | <file> | <webcam>`, `--simulate`,
  `--max-seconds`, `--loop`, `--no-display`, `--out`)
- `yolov8n.onnx` — YOLOv8n weights (12.8 MB, under GitHub's 25 MB
  web-upload cap). If missing, the script exports it once from
  `yolov8n.pt` via Ultralytics or prints the download link.
- `requirements.txt`, `.gitignore`
- Demo: `../../project-demos/seatbelt-compliance-detection-demo.mp4`
  (+ H.264 copy in `../../project-demos/linkedin-h264/`,
  codec-verified h264) with
  `seatbelt-compliance-detection-demo_events.log` and
  `seatbelt-compliance-detection-demo_summary.txt` alongside it.
- Raw footage: `../../project-demos/real-footage/raw/
  highway-approaching-cars-raw.mp4` (see `SOURCES.md`, entry 30).

## Run it

```bash
pip install -r requirements.txt
python seatbelt_compliance_detection.py --source road.mp4
python seatbelt_compliance_detection.py --simulate --max-seconds 30 --no-display
```

Press Q to quit. The video is written to `output.mp4` (or `--out`),
the timestamped event log to `output_events.log` and the final
counters to `output_summary.txt` next to it.

Author: Abdul Hanan — Computer Vision & AI Developer
