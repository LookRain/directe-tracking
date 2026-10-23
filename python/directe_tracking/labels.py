"""Human-checked identity labels: every N seconds, which box is the tagged climber (or none).

Stored in $DIRECTE_LABELS (labels/<clip>.json, small). Unlike the silver ground truth used by the
simulator, these catch identity mistakes, so they're what tracker changes are judged on.

Workflow: `propose` drafts labels from a tracker run (boxes snapped to detections); a person
checks each sample (lab peek / contact sheets) and records corrections as overrides
{time: detection number on that frame, None if the climber isn't in view, or UNDETECTED if they
are in view but the detector missed them}. Either way the right behaviour is to lock onto no one.
"""

from __future__ import annotations

import json

import numpy as np

import os
from pathlib import Path

from .clips import Clip
from .tracker import LOCKED, iou

LABEL_DIR = Path(os.environ.get("DIRECTE_LABELS", "labels"))  # <clip id>.json per labelled clip
UNDETECTED = "undetected"


def path(clip: Clip):
    return LABEL_DIR / f"{clip.id}.json"


def propose(clip: Clip, dets, rows, times, overrides: dict[float, int | None]) -> dict:
    by_i = {i: s for i, s in rows}
    idx = np.array(sorted(by_i))
    samples = []
    for t in times:
        i = int(idx[np.argmin(np.abs(dets.t[idx] - t))])
        sl = dets.frame(i)
        boxes = dets.boxes[sl]
        key = round(float(t), 1)
        visible = True
        if key in overrides:
            n = overrides[key]
            box = boxes[n] if isinstance(n, int) else None
            visible = n is not None
            source = "checked"
        else:
            s = by_i[i]
            box, source = None, "tracker-checked"
            if s.state == LOCKED and len(boxes):
                ov = iou(s.box, boxes)
                if ov.max() >= 0.5:
                    box = boxes[int(np.argmax(ov))]
        samples.append({"t": round(float(dets.t[i]), 2), "frame": i,
                        "box": None if box is None else [round(float(v), 1) for v in box],
                        "visible": bool(visible), "source": source})
    data = {"clip": clip.id, "target": clip.target, "samples": samples}
    LABEL_DIR.mkdir(exist_ok=True)
    path(clip).write_text(json.dumps(data, indent=1))
    return data


def load(clip: Clip) -> list[dict] | None:
    p = path(clip)
    return json.loads(p.read_text())["samples"] if p.exists() else None


def score(clip: Clip, rows) -> dict | None:
    """Identity accuracy of a tracker run against the labels."""
    samples = load(clip)
    if samples is None:
        return None
    by_i = {i: s for i, s in rows}
    c = {"right": 0, "wrong": 0, "missed": 0, "false_lock": 0, "absent_ok": 0}
    for smp in samples:
        s = by_i.get(smp["frame"])
        if s is None:
            continue
        locked = s.state == LOCKED
        if smp["box"] is None:
            c["false_lock" if locked else "absent_ok"] += 1
        elif not locked:
            c["missed"] += 1
        elif iou(s.box, np.array([smp["box"]]))[0] >= 0.3:
            c["right"] += 1
        else:
            c["wrong"] += 1
    visible = c["right"] + c["wrong"] + c["missed"]
    c["visible"] = visible
    c["identity_acc"] = round(c["right"] / max(c["right"] + c["wrong"], 1), 3)
    c["coverage"] = round(c["right"] / max(visible, 1), 3)
    return c
