#!/usr/bin/env python3
"""
Seatbelt Compliance Detection (traffic enforcement)
====================================================

A roadside / traffic-camera view of cars driving toward the camera.
YOLOv8n finds the vehicles, the windshield band of each vehicle box
is split into two seat regions (driver / passenger side), and a
diagonal-belt check across each occupant's torso gives a BELTED /
NO BELT verdict per seat:

* **Windshield band** -- the upper part of the vehicle box
  (x 0.10-0.90, y 0.06-0.42 of the box). It is a documented IMAGE
  REGION for this scene, not a calibrated camera model: on a
  front-facing car the windscreen sits in the upper part of the
  box, and the band is where the occupants' torsos are. A side-on
  car puts its windows lower in the box, so its seats read
  UNCLEAR rather than a wrong verdict (see the gates below).
* **Belt check** -- inside each seat's torso region (the lower 65%
  of the seat region) the pipeline runs CLAHE + Canny edges +
  Hough line segments and looks for a diagonal line (25-65 deg
  from horizontal, either direction) that spans at least 45% of
  the torso height. A 3-point belt runs shoulder-to-hip, so a
  genuine strap is one long diagonal; a window frame, a headrest
  edge or a reflection is short, horizontal or vertical. The
  check scores the total qualifying diagonal length against the
  torso diagonal: BELTED at >= 0.45, otherwise NO BELT.
* **Honesty gates** -- a verdict is only attempted when the seat
  region is big enough to judge (at least 40x35 px) and neither
  too dark (mean < 60) nor blown out (mean > 245) -- a dark,
  tinted or distant windscreen reads UNCLEAR, never NO BELT: the
  detector must not accuse a driver it cannot see. A perfectly
  uniform torso region also reads UNCLEAR: it is an empty seat
  (bare glass), not an occupant. A vehicle box clipped by the
  frame edge also reads UNCLEAR, because its windscreen is out
  of frame. There is deliberately no minimum-edge gate: a bright,
  featureless torso is exactly what an unbelted occupant looks
  like, so the absence of a diagonal strap in a clearly visible
  torso is a NO BELT verdict.
* **Confirmation** -- a seat verdict must repeat in 3 consecutive
  detection rounds before it counts and is logged, so one
  flickering YOLO box or one glinting reflection cannot raise a
  flag. A NO BELT verdict is a flag FOR HUMAN REVIEW, never a
  fine: the log says so on every entry.

The compliance board shows vehicles tracked, seats assessed,
BELTED / NO BELT / UNCLEAR counts and the compliance rate among
decided seats (belted / (belted + no belt)).

Weights: ``yolov8n.onnx`` next to this script (12.8 MB, ships in
this folder). If it is missing, the script tries to export it from
``yolov8n.pt`` with Ultralytics (downloading the .pt if needed) and
otherwise exits with instructions.

Simulation mode (``--simulate``) renders a roadside view with
three cars driving toward the camera: car 1 has a belted driver
and a belted passenger, car 2 a beltless driver and a belted
passenger, car 3 a beltless driver and an empty passenger seat.
The simulation's vehicle source finds the cars with a colour
mask + contours -- the windshield band, the belt check and the
verdict logic are the exact same code as real mode, and the
detector never sees the simulation's ground truth. Simulation is
an internal logic check; the delivered demo is real footage
(see README).

Honest framing: a strap is hard to see through tinted glass and
at distance, and a windscreen reflection can hide it entirely.
The detector flags a missing strap for human review, never a
fine, and on ordinary roadside footage most seats read UNCLEAR
-- that is the honest result, not a failure to detect.

Examples
--------
    python seatbelt_compliance_detection.py                 (file picker)
    python seatbelt_compliance_detection.py --source pick
    python seatbelt_compliance_detection.py --source road.mp4
    python seatbelt_compliance_detection.py --source 0      (webcam)
    python seatbelt_compliance_detection.py --simulate --max-seconds 30 --no-display

Press Q to quit. The video is written to ``output.mp4``, the
timestamped event log to ``output_events.log`` and the final
counters to ``output_summary.txt`` next to it.

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import urllib.request

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Layout (the output video is always 1280x720)
# ---------------------------------------------------------------------------
W, H = 1280, 720
TITLE_H = 46
PANEL_H = 150                     # stats panel at the bottom
VIDEO_W, VIDEO_H = W, H - TITLE_H - PANEL_H
BG = (24, 26, 32)
PANEL_BG = (32, 35, 46)
WHITE = (255, 255, 255)
GREY = (160, 166, 178)
GREEN = (90, 200, 90)
AMBER = (60, 170, 255)
RED = (60, 60, 235)
YELLOW = (0, 210, 235)

# ---------------------------------------------------------------------------
# Detection + belt-check tuning
# ---------------------------------------------------------------------------
DET_W, DET_H = 1280, 720          # detection runs on this frame size
DETECT_EVERY = 3                  # run YOLO every 3rd frame (10 fps)
CONF = 0.30
NMS_THRESH = 0.30
VEHICLE_CLASSES = {2, 3, 5, 7}    # car, motorcycle, bus, truck (COCO)
MATCH_DIST = 80                   # px, centroid match radius for small
                                  # (distant) vehicles; see the
                                  # scale-aware gate in update()
MAX_MISSED = 15                   # detection rounds before a track drops
# Windshield band, as fractions of the vehicle box. On a front-facing
# car the windscreen sits in the upper part of the box; the band is
# an image region for this scene, not a calibrated camera model.
SHIELD = (0.10, 0.06, 0.90, 0.42)  # x0, y0, x1, y1 (fractions of box)
TORSO_TOP_FRAC = 0.35             # torso = lower 65% of the seat region
# A seat is only judged when the vehicle is big enough: a car box
# narrower than this puts each seat under ~40 px, too small for a
# strap to be readable.
MIN_VEHICLE_W = 110
MIN_SEAT_W, MIN_SEAT_H = 40, 35
MIN_MEAN_BRIGHTNESS = 60.0        # darker than this: UNCLEAR, not NO BELT
# Raised from 30: on the demo clip the seat regions of dark
# windscreens read mean 39-58, and a strap is invisible to a human
# in pixels that dark too -- at 30 they were all judged NO BELT
# (diagonal 0.00) and the board read 0% compliance, which overstates
# the finding. At 60 the dark-glass seats read UNCLEAR and only
# seats bright enough to judge (mean 61-98 on this clip) are
# decided. The simulated torso reads ~197, far above the gate.
MAX_MEAN_BRIGHTNESS = 245.0       # brighter than this: blown out, UNCLEAR
BELT_MIN_ANGLE = 20.0             # deg from horizontal (either diagonal)
# Lowered from 25 deg: the torso region is wider than it is tall
# (about 2:1 on the demo geometry), so a shoulder-to-hip strap
# crosses it at ~23 deg -- at 25 deg the simulated belt (drawn
# across the whole torso, both edges found by Hough) scored 0.00
# and every belted seat read NO BELT. Window frames and headrest
# edges are near 0 or 90 deg, so 20 deg still excludes them.
BELT_MAX_ANGLE = 65.0
BELT_SCORE_MIN = 0.45             # qualifying diagonal length / torso diag
BELT_SPAN_MIN = 0.45              # longest segment must span this much
                                  # of the torso height
CONFIRM_ROUNDS = 3                # a verdict must repeat this often
                                  # before it counts and is logged
PT_URL = ("https://github.com/ultralytics/assets/releases/"
          "download/v8.2.0/yolov8n.pt")

SEAT_NAMES = ("A", "B")           # left / right seat of the windscreen


# ---------------------------------------------------------------------------
# Vehicle sources -- real mode (YOLOv8 via OpenCV DNN) and simulation
# (colour mask). Both return boxes in detection-frame coordinates.
# ---------------------------------------------------------------------------
def ensure_weights(path):
    """Return a usable yolov8n.onnx path, exporting it if needed."""
    if os.path.exists(path):
        return path
    print(f"Weights not found at {path}; trying Ultralytics export ...")
    try:
        from ultralytics import YOLO
        pt = os.path.join(os.path.dirname(path) or ".", "yolov8n.pt")
        if not os.path.exists(pt):
            print("Downloading yolov8n.pt ...")
            urllib.request.urlretrieve(PT_URL, pt)
        print("Exporting yolov8n.pt -> yolov8n.onnx (one-time) ...")
        YOLO(pt).export(format="onnx", imgsz=640)
        exported = os.path.join(os.path.dirname(pt) or ".", "yolov8n.onnx")
        if os.path.exists(exported) and exported != path:
            os.replace(exported, path)
        return path
    except Exception as exc:  # noqa: BLE001 - report and stop cleanly
        print(f"Could not obtain weights: {exc}")
        print("Download yolov8n.onnx and place it next to this script, or")
        print("install ultralytics (`pip install ultralytics`) and re-run.")
        sys.exit(1)


class YoloVehicleDetector:
    """YOLOv8n vehicle detection through OpenCV DNN."""

    def __init__(self, weights_path):
        self.net = cv2.dnn.readNetFromONNX(ensure_weights(weights_path))

    def detect(self, frame):
        """frame is a DET_W x DET_H image; returns [x, y, w, h, conf]."""
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()[0].T  # [8400, 84]: cx cy w h + 80 scores
        scores = out[:, 4:].max(axis=1)
        classes = out[:, 4:].argmax(axis=1)
        keep = (scores > CONF) & np.isin(classes, list(VEHICLE_CLASSES))
        sh, sw = frame.shape[:2]
        boxes = []
        for (cx, cy, bw, bh), s in zip(out[keep, :4], scores[keep]):
            boxes.append([float((cx - bw / 2) * sw / 640),
                          float((cy - bh / 2) * sh / 640),
                          float(bw * sw / 640), float(bh * sh / 640),
                          float(s)])
        if not boxes:
            return []
        idx = cv2.dnn.NMSBoxes([b[:4] for b in boxes],
                               [b[4] for b in boxes], CONF, NMS_THRESH)
        return [boxes[int(i)] for i in idx.flatten()] if len(idx) else []


def sim_vehicle_boxes(frame):
    """Colour-blob vehicle source for the rendered roadside view."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([95, 120, 80]),
                       np.array([115, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < 400:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([float(x), float(y), float(w), float(h), 1.0])
    return boxes


# ---------------------------------------------------------------------------
# Belt check -- the same function runs on real and simulated pixels
# ---------------------------------------------------------------------------
def seat_rects(box):
    """Two seat rectangles (left/right) inside the windshield band.

    Returns [(x0, y0, x1, y1), ...] in detection-frame coordinates,
    or [] when the vehicle is too small or its box is clipped by
    the frame edge (the windscreen is then out of frame).
    """
    x, y, w, h, _ = box
    if w < MIN_VEHICLE_W:
        return []
    # A box touching the frame edge has its windscreen cut off: the
    # band would be measured on a partial car, so do not judge it.
    if x <= 2 or y <= 2 or x + w >= DET_W - 2 or y + h >= DET_H - 2:
        return []
    sx0, sy0, sx1, sy1 = SHIELD
    bx0, by0 = x + sx0 * w, y + sy0 * h
    bx1, by1 = x + sx1 * w, y + sy1 * h
    mid = (bx0 + bx1) / 2.0
    return [(bx0, by0, mid, by1), (mid, by0, bx1, by1)]


def belt_check(gray, rect):
    """Verdict for one seat: 'BELTED', 'NO BELT' or 'UNCLEAR'.

    ``gray`` is the greyscale detection frame; ``rect`` is the seat
    rectangle inside the windshield band. The torso region is the
    lower 65% of the seat. Returns (verdict, score, detail) where
    detail is a short string for the event log / debugging.
    """
    x0, y0, x1, y1 = [int(v) for v in rect]
    # Torso: lower 65% of the seat, inset 8% on the sides.
    ty0 = int(y0 + TORSO_TOP_FRAC * (y1 - y0))
    tx0 = int(x0 + 0.08 * (x1 - x0))
    tx1 = int(x1 - 0.08 * (x1 - x0))
    tw, th = tx1 - tx0, y1 - ty0
    if tw < MIN_SEAT_W or th < MIN_SEAT_H:
        return "UNCLEAR", 0.0, "torso too small"
    roi = gray[ty0:y1, tx0:tx1]
    if roi.size == 0:
        return "UNCLEAR", 0.0, "empty torso"
    mean = float(roi.mean())
    if mean < MIN_MEAN_BRIGHTNESS:
        return "UNCLEAR", 0.0, f"too dark (mean {mean:.0f})"
    if mean > MAX_MEAN_BRIGHTNESS:
        # Blown out (sky glare on the glass): nothing readable.
        return "UNCLEAR", 0.0, f"blown out (mean {mean:.0f})"
    std = float(roi.std())
    if std < 6.0:
        # A perfectly uniform torso region is an empty seat (bare
        # glass) or a windscreen with nothing readable in it -- not
        # an occupant. An occupied torso always has some texture
        # (clothing folds, shading, a collar). Without this gate
        # the simulation's empty passenger seat read NO BELT: its
        # dark glass (mean ~41) cleared the brightness gate and
        # "no diagonal" became a false accusation.
        return "UNCLEAR", 0.0, f"uniform (std {std:.1f})"
    # NOTE: there is deliberately NO minimum edge-density gate here.
    # An earlier build returned UNCLEAR whenever the torso had
    # almost no edges -- but a bright, featureless torso is exactly
    # what an unbelted occupant looks like, so every NO BELT seat
    # read UNCLEAR instead and the belted/unbelted split never
    # appeared in simulation. If the torso is bright enough to see
    # and big enough to judge, the absence of a diagonal strap is a
    # NO BELT verdict (a flag for human review, never a fine). A
    # strap the same colour as the clothing can still read NO BELT
    # -- that limit is stated in the README.
    edges = cv2.Canny(roi, 50, 150)
    diag = math.hypot(tw, th)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=22,
                            minLineLength=int(0.30 * diag), maxLineGap=10)
    total = 0.0
    longest_span = 0.0
    if lines is not None:
        # OpenCV 5 returns HoughLinesP as (N, 4); older versions as
        # (N, 1, 4) -- reshape first so the unpack below works on both.
        for x_a, y_a, x_b, y_b in lines.reshape(-1, 4):
            dx, dy = abs(x_b - x_a), abs(y_b - y_a)
            length = math.hypot(dx, dy)
            if dx == 0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if not (BELT_MIN_ANGLE <= angle <= BELT_MAX_ANGLE):
                continue
            total += length
            longest_span = max(longest_span, dy)
    score = total / diag if diag > 0 else 0.0
    if score >= BELT_SCORE_MIN and longest_span >= BELT_SPAN_MIN * th:
        return "BELTED", score, f"diagonal {score:.2f}"
    return "NO BELT", score, f"diagonal {score:.2f}"


# ---------------------------------------------------------------------------
# Seatbelt tracker -- centroid tracking + per-seat confirmed verdicts
# ---------------------------------------------------------------------------
class SeatbeltTracker:
    """Tracks vehicles and watches the belt verdict of each seat."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.frames = 0
        self.video_frames = 0
        self.events = []
        self.detections = 0
        self.vehicles_seen = 0
        self._last_t = None
        # Counted seat outcomes, keyed by (vehicle id, seat): the
        # verdict a seat counts as. A seat counts its FIRST decided
        # verdict (BELTED / NO BELT); a seat that first reads
        # UNCLEAR while sliding into frame and is later decided
        # counts the decided verdict, not the early UNCLEAR -- an
        # earlier build counted only the first-ever confirmation,
        # so most seats counted UNCLEAR (they were unreadable
        # during entry) and the board read 1 BELTED / 1 NO BELT /
        # 8 UNCLEAR while the log showed three BELTED
        # confirmations. A decided seat that later goes unreadable
        # keeps its decided outcome (the log still records the
        # change); a flip between decided verdicts moves the count.
        self.seat_outcomes = {}
        self.no_belt_events = 0
        self.first_no_belt_t = None

    @property
    def belted(self):
        return sum(1 for v in self.seat_outcomes.values()
                   if v == "BELTED")

    @property
    def no_belt(self):
        return sum(1 for v in self.seat_outcomes.values()
                   if v == "NO BELT")

    @property
    def unclear(self):
        return sum(1 for v in self.seat_outcomes.values()
                   if v == "UNCLEAR")

    def _log(self, t, text):
        self.events.append((t, text))

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah, _ = a
        bx, by, bw, bh, _ = b
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _centre(box):
        x, y, w, h, _ = box
        return (x + w / 2, y + h / 2)

    def reset(self, t):
        """Drop all tracks: the source looped, so the scene restarted."""
        n = len(self.tracks)
        self.tracks.clear()
        self._log(t, f"source looped -- {n} in-flight tracks reset "
                     f"(seat verdicts restart from zero)")

    @staticmethod
    def _new_seat():
        return {"pending": None, "streak": 0, "confirmed": None,
                "score": 0.0, "detail": ""}

    def update(self, boxes, gray, t):
        """One detection round. Returns the live state dict."""
        self.frames += 1
        self.detections += len(boxes)
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        used = set()
        for box in boxes:
            cx, cy = self._centre(box)
            best, best_d = None, float("inf")
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                d = math.hypot(cx - tr["cx"], cy - tr["cy"])
                # Scale-aware match gate: a car driving straight at
                # the camera grows fast -- on the demo clip the
                # approaching Fiat's centroid jumps ~95 px between
                # detection rounds and its box IoU falls to 0.046,
                # so the fixed 80 px gate (borrowed from the
                # blind-spot tracker, where cars slide sideways
                # slowly) spawned a new track for the same car every
                # few rounds: 32 "vehicles" for ~10 real cars, and
                # each fragment re-counted the same seats. The gate
                # is half the bigger box's size, so a big close car
                # may jump further than a small distant one, while
                # two cars in neighbouring lanes (centroids hundreds
                # of px apart) still cannot steal each other's track.
                gate = max(MATCH_DIST,
                           0.5 * max(box[2], box[3],
                                     tr["box"][2], tr["box"][3]))
                if d >= gate:
                    continue
                if d >= best_d:
                    continue
                # A detection must also OVERLAP the track's last box.
                # The threshold is low (0.02) for the same reason:
                # a fast-approaching box overlaps its predecessor
                # by only ~0.05, while two different cars' boxes
                # essentially never overlap at all.
                if self._iou(box, tr["box"]) <= 0.02:
                    continue
                best, best_d = tid, d
            if best is None:
                tid = self.next_id
                self.next_id += 1
                self.vehicles_seen += 1
                self.tracks[tid] = {"id": tid, "cx": cx, "cy": cy,
                                    "box": box, "missed": 0,
                                    "seats": {name: self._new_seat()
                                              for name in SEAT_NAMES}}
                best = tid
            used.add(best)
            tr = self.tracks[best]
            tr.update({"cx": cx, "cy": cy, "box": box, "missed": 0})
            # Belt check per seat, on this frame's pixels.
            rects = seat_rects(box)
            for i, name in enumerate(SEAT_NAMES):
                seat = tr["seats"][name]
                if i >= len(rects):
                    verdict, score, detail = (
                        "UNCLEAR", 0.0, "vehicle too small / clipped")
                else:
                    verdict, score, detail = belt_check(gray, rects[i])
                if verdict == seat["pending"]:
                    seat["streak"] += 1
                else:
                    seat["pending"] = verdict
                    seat["streak"] = 1
                seat["score"] = score
                seat["detail"] = detail
                if (seat["streak"] >= CONFIRM_ROUNDS
                        and seat["confirmed"] != verdict):
                    prev = seat["confirmed"]
                    seat["confirmed"] = verdict
                    key = (tid, name)
                    prev_outcome = self.seat_outcomes.get(key)
                    if verdict in ("BELTED", "NO BELT"):
                        if prev_outcome != verdict:
                            if verdict == "NO BELT":
                                self.no_belt_events += 1
                                if self.first_no_belt_t is None:
                                    self.first_no_belt_t = t
                            self.seat_outcomes[key] = verdict
                    elif prev_outcome is None:
                        # First confirmation is UNCLEAR: record it
                        # only so the seat counts as assessed; a
                        # later decided verdict overwrites it (see
                        # above) and the board never shows a seat
                        # as UNCLEAR once it has been decided.
                        self.seat_outcomes[key] = "UNCLEAR"
                    if verdict == "NO BELT":
                        self._log(t, f"vehicle #{tid} seat {name} "
                                     f"NO BELT -- flagged for human "
                                     f"review ({detail})")
                    elif verdict == "BELTED":
                        self._log(t, f"vehicle #{tid} seat {name} "
                                     f"BELTED ({detail})")
                    elif prev is not None:
                        # A seat that was decided and is now
                        # unreadable (car turning away, glass
                        # glinting) must not silently keep its
                        # old verdict on the board.
                        self._log(t, f"vehicle #{tid} seat {name} "
                                     f"now UNCLEAR (was {prev})")
        for tid in list(self.tracks):
            if tid in used:
                continue
            tr = self.tracks[tid]
            tr["missed"] += 1
            if tr["missed"] > MAX_MISSED:
                live = [f"seat {n} {s['confirmed']}"
                        for n, s in tr["seats"].items()
                        if s["confirmed"] == "NO BELT"]
                if live:
                    self._log(t, f"vehicle #{tid} left the frame "
                                 f"({', '.join(live)} flagged)")
                del self.tracks[tid]
        # Live board state.
        any_no_belt = any(
            s["confirmed"] == "NO BELT"
            for tr in self.tracks.values()
            for s in tr["seats"].values())
        decided = self.belted + self.no_belt
        compliance = (100.0 * self.belted / decided) if decided else None
        return {"alert": ("NO BELT FLAGGED"
                          if any_no_belt else "MONITORING"),
                "any_no_belt": any_no_belt,
                "compliance": compliance}

    def summary(self):
        decided = self.belted + self.no_belt
        lines = [
            "SEATBELT COMPLIANCE DETECTION -- RUN SUMMARY",
            f"frames processed: {self.video_frames}",
            f"tracker updates: {self.frames}",
            f"vehicle detections (summed): {self.detections}",
            f"vehicles tracked: {self.vehicles_seen}",
            f"seats assessed (confirmed verdicts): "
            f"{self.belted + self.no_belt + self.unclear}",
            f"  BELTED: {self.belted}",
            f"  NO BELT (flagged for human review): {self.no_belt}",
            f"  UNCLEAR: {self.unclear}",
            ("compliance among decided seats: "
             + (f"{100.0 * self.belted / decided:.1f}% "
                f"({self.belted}/{decided})" if decided
                else "no decided seats")),
            f"NO BELT events logged: {self.no_belt_events}",
            ("first NO BELT at: "
             + (f"{self.first_no_belt_t:.1f} s"
                if self.first_no_belt_t is not None else "never")),
            f"events logged: {len(self.events)}",
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Simulation renderer -- a roadside view; cars drive toward the camera
# with belted and unbelted occupants. The tracker only ever sees
# these pixels.
# ---------------------------------------------------------------------------
class RoadsideSim:
    """Renders cars approaching on a road, occupants in the windscreen.

    The drawn cars use the same geometry the pipeline assumes: the
    windshield is x 0.10-0.90, y 0.06-0.42 of the car box, and each
    torso is the lower 65% of its seat half. A belted occupant gets
    a light diagonal strap across the torso; an unbelted one gets
    none. The strap is drawn on the torso pixels, so the belt check
    has to find it in the rendered image -- it never sees the
    script's ground truth.
    """

    CAR_BLUE = (255, 130, 30)     # crowd-density blue: the vehicle mask
    CAR_DARK = (200, 100, 20)     # darker blue, still inside the mask
    GLASS = (38, 40, 46)          # windscreen: dark, outside the mask,
                                  # fully enclosed by the blue body so
                                  # the outer contour stays the car
    SKIN = (170, 190, 220)        # occupant head / torso (light, so the
                                  # belt line contrasts against it)
    SHIRT_DARK = (140, 160, 190)  # collar / seam lines on the torso
    BELT = (245, 245, 245)        # the strap itself

    def __init__(self, w, h, fps=30):
        self.w, self.h = w, h
        self.fps = fps
        self.frame_idx = 0
        # (centre-x frac, enter t, leave t, width px, height px,
        #  driver belted?, passenger belted?/None for empty seat).
        # The cars are staged so a sliding car never crosses another
        # car's position: car 2 leaves (14 s) before car 3 enters
        # (16 s). An earlier staging had car 2 sliding out right
        # straight through car 3, the two blue blobs merged into one
        # contour, and the tracker spawned fragment tracks with
        # verdicts flipping between the two windscreens -- found on
        # the third simulation run.
        self.cars = [
            {"cx": 0.22, "enter": 0.0, "leave": 99.0,
             "w": 300, "h": 210, "seats": (True, True)},
            {"cx": 0.52, "enter": 3.0, "leave": 14.0,
             "w": 300, "h": 210, "seats": (False, True)},
            {"cx": 0.80, "enter": 16.0, "leave": 28.0,
             "w": 300, "h": 210, "seats": (False, None)},
        ]

    def _draw_road(self, img):
        cv2.rectangle(img, (0, 0), (self.w, int(self.h * 0.45)),
                      (120, 140, 165), -1)
        cv2.rectangle(img, (0, int(self.h * 0.45)), (self.w, self.h),
                      (70, 72, 76), -1)
        for x in range(0, self.w, 90):
            cv2.circle(img, (x + 20, int(self.h * 0.42)), 24,
                       (60, 110, 60), -1)
        for i in range(5):
            y = int(self.h * (0.55 + i * 0.09))
            cv2.rectangle(img, (self.w // 2 - 6, y),
                          (self.w // 2 + 6, y + 22), (220, 220, 210), -1)

    def _draw_car(self, img, cx, bottom, w, h, seats):
        x0, y0 = int(cx - w / 2), int(bottom - h)
        x1, y1 = int(cx + w / 2), int(bottom)
        cv2.rectangle(img, (x0, y0), (x1, y1), self.CAR_BLUE, -1)
        # Roof band, still inside the mask range.
        cv2.rectangle(img, (x0, y0), (x1, y0 + int(h * 0.06)),
                      self.CAR_DARK, -1)
        # Windshield: x 0.10-0.90, y 0.06-0.42 of the car box, exactly
        # the band the pipeline reads.
        wx0 = int(x0 + 0.10 * w)
        wx1 = int(x0 + 0.90 * w)
        wy0 = int(y0 + 0.06 * h)
        wy1 = int(y0 + 0.42 * h)
        cv2.rectangle(img, (wx0, wy0), (wx1, wy1), self.GLASS, -1)
        # Headlights and grille, below the windscreen.
        cv2.rectangle(img, (x0 + 12, y1 - 34), (x0 + 62, y1 - 14),
                      (230, 230, 200), -1)
        cv2.rectangle(img, (x1 - 62, y1 - 34), (x1 - 12, y1 - 14),
                      (230, 230, 200), -1)
        cv2.rectangle(img, (x0 + int(w * 0.35), y1 - 30),
                      (x1 - int(w * 0.35), y1 - 12), (40, 40, 44), -1)
        # Occupants: one per seat half. Head circle above, torso
        # rectangle in the lower 65% of the seat -- the same torso
        # the belt check reads.
        mid = (wx0 + wx1) // 2
        for i, belted in enumerate(seats):
            if belted is None:
                continue  # empty seat: no occupant drawn at all
            sx0 = wx0 if i == 0 else mid
            sx1 = mid if i == 0 else wx1
            scx = (sx0 + sx1) // 2
            seat_h = wy1 - wy0
            head_r = max(6, int(seat_h * 0.13))
            head_cy = wy0 + int(seat_h * 0.22)
            cv2.circle(img, (scx, head_cy), head_r, self.SKIN, -1)
            ty0 = wy0 + int(seat_h * TORSO_TOP_FRAC)
            cv2.rectangle(img, (sx0 + 6, ty0), (sx1 - 6, wy1),
                          self.SKIN, -1)
            # Clothing texture: a horizontal collar line and a
            # vertical centre seam. Both are axis-aligned, so the
            # belt check's diagonal gate ignores them -- but they
            # keep an occupied torso from reading as a perfectly
            # uniform (empty) region. Without them an unbelted
            # occupant in simulation is a flat patch and the
            # uniform-region gate above would call it UNCLEAR.
            cv2.line(img, (sx0 + 6, ty0 + 8), (sx1 - 6, ty0 + 8),
                     self.SHIRT_DARK, 3)
            cv2.line(img, (scx, ty0), (scx, wy1), self.SHIRT_DARK, 2)
            if belted:
                # The strap: shoulder to opposite hip, a thick
                # light diagonal across the whole torso.
                cv2.line(img, (sx0 + 8, ty0 + 2),
                         (sx1 - 8, wy1 - 2), self.BELT, 5)

    def frame(self):
        idx = self.frame_idx
        self.frame_idx += 1
        t = idx / self.fps
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._draw_road(img)
        for car in self.cars:
            if t < car["enter"] or t > car["leave"] + 1.0:
                continue
            # Slide in from the NEAREST frame edge over 1 s, then
            # hold; the car also grows slightly as it approaches.
            # (An earlier build slid every car in from the left, so
            # car 2 drove straight through car 1's box, the two blue
            # blobs merged into one contour, and the tracker spawned
            # duplicate tracks -- found on the first simulation run.)
            from_left = car["cx"] < 0.5
            edge_cx = -0.25 if from_left else 1.25
            if t < car["enter"] + 1.0:
                f = (t - car["enter"]) / 1.0
                cx = edge_cx + f * (car["cx"] - edge_cx)
                scale = 0.9
            elif t <= car["leave"]:
                cx = car["cx"]
                f = (t - car["enter"]) / max(1.0, car["leave"]
                                             - car["enter"])
                scale = 0.9 + 0.15 * f
            else:
                f = (t - car["leave"]) / 1.0
                cx = car["cx"] + f * (edge_cx - car["cx"])
                scale = 1.05
            w = int(car["w"] * scale)
            h = int(car["h"] * scale)
            self._draw_car(img, int(cx * self.w), int(self.h * 0.86),
                           w, h, car["seats"])
        return img


# ---------------------------------------------------------------------------
# Rendering -- draws tracker output onto the 1280x720 canvas
# ---------------------------------------------------------------------------
def draw_canvas(det_frame, gray, tracker, res, elapsed, mode_label, boxes):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = BG
    cv2.rectangle(canvas, (0, 0), (W, TITLE_H), PANEL_BG, -1)
    cv2.putText(canvas, "SEATBELT COMPLIANCE DETECTION  --  traffic "
                        "enforcement",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
    cv2.putText(canvas, mode_label, (W - 260, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)

    vh, vw = det_frame.shape[:2]
    scale = min(VIDEO_W / vw, VIDEO_H / vh)
    nw, nh = int(vw * scale), int(vh * scale)
    resized = cv2.resize(det_frame, (nw, nh))
    ox = (VIDEO_W - nw) // 2
    oy = TITLE_H + (VIDEO_H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    sx, sy = nw / vw, nh / vh

    def _mx(x):
        return ox + int(x * sx)

    def _my(y):
        return oy + int(y * sy)

    # Vehicle boxes, windshield bands and per-seat verdicts.
    for tid, tr in tracker.tracks.items():
        # A track YOLO has not seen for 3 detection rounds (0.3 s)
        # is stale: do not draw its last box.
        if tr["missed"] > 2:
            continue
        x, y, w, h, _ = tr["box"]
        any_no_belt = any(s["confirmed"] == "NO BELT"
                          for s in tr["seats"].values())
        colour = RED if any_no_belt else GREY
        cv2.rectangle(canvas, (_mx(x), _my(y)), (_mx(x + w), _my(y + h)),
                      colour, 3 if any_no_belt else 2)
        # Windshield band.
        sx0, sy0, sx1, sy1 = SHIELD
        wx0, wy0 = _mx(x + sx0 * w), _my(y + sy0 * h)
        wx1, wy1 = _mx(x + sx1 * w), _my(y + sy1 * h)
        cv2.rectangle(canvas, (wx0, wy0), (wx1, wy1), YELLOW, 1)
        # Seat verdicts are drawn as coloured seat outlines only.
        # The verdict text lives in the single vehicle tag below:
        # an earlier build printed one tag per seat at the top of
        # each seat rectangle, and the two tags (each wider than its
        # seat) printed on top of each other into unreadable mush,
        # and clipped off the top of the video area on small cars.
        # The tag shows CONFIRMED verdicts only: an earlier build
        # showed the pending (this-round) verdict, so a seat could
        # read BELTED in the tag while the board still counted 0
        # BELTED -- the verdict had not yet survived the 3-round
        # confirmation. An undecided seat reads "--".
        rects = seat_rects(tr["box"])
        for i, name in enumerate(SEAT_NAMES):
            seat = tr["seats"][name]
            verdict = seat["confirmed"]
            if verdict is None or i >= len(rects):
                continue
            vcolour = (GREEN if verdict == "BELTED" else
                       RED if verdict == "NO BELT" else GREY)
            rx0, ry0, rx1, ry1 = rects[i]
            cv2.rectangle(canvas, (_mx(rx0), _my(ry0)),
                          (_mx(rx1), _my(ry1)), vcolour, 1)
        # Vehicle tag above the box: id + both seat verdicts on one
        # line, so nothing overlaps.
        tag = f"#{tid}"
        verdict_bits = []
        for name in SEAT_NAMES:
            seat = tr["seats"][name]
            verdict = seat["confirmed"]
            if verdict is None:
                verdict_bits.append(f"{name}: --")
                continue
            bit = f"{name}: {verdict}"
            if verdict in ("BELTED", "NO BELT"):
                bit += f" {seat['score']:.2f}"
            verdict_bits.append(bit)
        if verdict_bits:
            tag += "  " + "  |  ".join(verdict_bits)
        if any_no_belt:
            tag += "  NO BELT -- REVIEW"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lx = _mx(x)
        if lx + tw + 8 > W - 4:
            lx = max(4, W - 4 - tw - 8)
        ly = _my(y) - 8
        if ly - th < oy + 4:
            ly = _my(y + h) + th + 10
        cv2.rectangle(canvas, (lx, ly - th - 4),
                      (lx + tw + 8, ly + 4), (24, 26, 32), -1)
        cv2.putText(canvas, tag, (lx + 4, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    # Bottom panel: board + alert banner.
    py = TITLE_H + VIDEO_H
    cv2.rectangle(canvas, (0, py), (W, H), PANEL_BG, -1)
    alert = res["alert"]
    acol = RED if res["any_no_belt"] else GREEN
    cv2.rectangle(canvas, (0, py), (W, py + 6), acol, -1)
    banner = f"SEATBELT: {alert}"
    if res["any_no_belt"]:
        banner += "  --  HUMAN REVIEW ONLY, NOT A FINE"
    cv2.putText(canvas, banner, (16, py + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, acol, 2)
    decided = tracker.belted + tracker.no_belt
    comp = (f"{100.0 * tracker.belted / decided:.0f}%"
            if decided else "--")
    cv2.putText(canvas, f"t = {elapsed:.1f} s", (16, py + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(canvas,
                f"compliance: {comp}   BELTED: {tracker.belted}   "
                f"NO BELT: {tracker.no_belt}   UNCLEAR: {tracker.unclear}",
                (300, py + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(canvas,
                f"vehicles: {tracker.vehicles_seen}   "
                f"seats assessed: "
                f"{tracker.belted + tracker.no_belt + tracker.unclear}   "
                f"NO BELT events: {tracker.no_belt_events}   "
                f"detections: {tracker.detections}",
                (16, py + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)
    return canvas


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        root.destroy()
        return path
    except Exception:
        return ""


def open_source(source):
    if source == "pick":
        path = pick_file()
        if not path:
            print("No file chosen.")
            sys.exit(0)
        source = path
    cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        sys.exit(1)
    return cap


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--source", default="pick",
                    help="video file, webcam index, or 'pick'")
    ap.add_argument("--simulate", action="store_true",
                    help="rendered roadside view instead of a camera/video")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--loop", action="store_true",
                    help="loop a short source until --max-seconds is reached")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--weights", default=None,
                    help="path to yolov8n.onnx (default: next to the script)")
    args = ap.parse_args()

    fps = 30
    here = os.path.dirname(os.path.abspath(__file__))
    detector = None if args.simulate else YoloVehicleDetector(
        args.weights or os.path.join(here, "yolov8n.onnx"))
    tracker = SeatbeltTracker()
    sim = RoadsideSim(DET_W, DET_H, fps) if args.simulate else None
    cap = None if args.simulate else open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else fps
    if not src_fps or src_fps <= 0:
        src_fps = fps
    step = max(1, round(src_fps / fps))  # sample the source at ~30 fps

    out_path = os.path.abspath(args.out)
    base, _ = os.path.splitext(out_path)
    events_path = base + "_events.log"
    summary_path = base + "_summary.txt"
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    mode_label = "SIMULATION (logic check)" if args.simulate else "REAL MODE"

    n_out = 0
    max_frames = int(args.max_seconds * fps)
    src_idx = 0
    boxes = []
    res = {"alert": "MONITORING", "any_no_belt": False,
           "compliance": None}
    while n_out < max_frames:
        t = n_out / fps
        if args.simulate:
            raw = sim.frame()
            det_frame = raw
            gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
            boxes = sim_vehicle_boxes(det_frame)
            res = tracker.update(boxes, gray, t)
        else:
            ok, frame = cap.read()
            src_idx += 1
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    src_idx = 1
                    tracker.reset(t)
                if not ok:
                    break
            if (src_idx - 1) % step != 0:
                continue
            det_frame = cv2.resize(frame, (DET_W, DET_H))
            gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
            if n_out % DETECT_EVERY == 0:
                boxes = detector.detect(det_frame)
                res = tracker.update(boxes, gray, t)
            else:
                res = {"alert": res["alert"],
                       "any_no_belt": any(
                           s["confirmed"] == "NO BELT"
                           for tr in tracker.tracks.values()
                           for s in tr["seats"].values()),
                       "compliance": res["compliance"]}
        canvas = draw_canvas(det_frame, gray, tracker, res, t,
                             mode_label, boxes)
        writer.write(canvas)
        n_out += 1
        if not args.no_display:
            try:
                cv2.imshow("Seatbelt Compliance Detection", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
            except cv2.error:
                pass

    writer.release()
    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    tracker.video_frames = n_out
    with open(events_path, "w") as f:
        for t, text in tracker.events:
            f.write(f"{t:6.1f}s  {text}\n")
    with open(summary_path, "w") as f:
        f.write(tracker.summary())
    print(tracker.summary())
    print(f"video:   {out_path}")
    print(f"events:  {events_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
