"""Hands-free registration: the climber holds an "L" pose to tell the camera "film me".

L pose: one arm straight up, the other straight out to the side, standing on straight legs,
held still for HOLD_S seconds. Chosen because it rarely happens by accident:
  - arms-up stretching at the base is a "Y" or "I", not an L;
  - on the wall, climbers often have one arm up and one out, but never on straight vertical
    legs while staying put for 1.5 s.
The checks are symmetric, so it works facing the camera or with the back turned (pose models
mix up left and right on back views anyway).

On a trigger the app would confirm visibly from a distance: blink the torch twice, play a sound,
speak the profile ("Locked on: white tee, orange shorts").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracker import iou

KP_CONF = 0.35
HOLD_S = 1.5
HOLD_FRACTION = 0.8      # of frames in the hold window that must show the pose
MAX_DRIFT = 0.3          # box-centre movement during the hold, in body heights
COOLDOWN_S = 3.0

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between vectors in degrees."""
    c = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _straight(p: np.ndarray, joint: np.ndarray, q: np.ndarray, min_deg: float = 145) -> bool:
    return _angle(p - joint, q - joint) >= min_deg


def l_pose(kpts: np.ndarray) -> bool:
    """True if this person is in the L pose (single frame)."""
    xy, ok = kpts[:, :2], kpts[:, 2] >= KP_CONF
    if not ok[[L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP]].all():
        return False
    torso = float(np.linalg.norm((xy[L_SH] + xy[R_SH]) / 2 - (xy[L_HIP] + xy[R_HIP]) / 2))
    if torso < 8:
        return False

    def arm(sh, el, wr, other_sh):
        v = xy[wr] - xy[sh]
        if not _straight(xy[sh], xy[el], xy[wr]) or np.linalg.norm(v) < 0.55 * torso:
            return None
        if _angle(v, np.array([0.0, -1.0])) <= 30:
            return "up"
        outward = np.sign(v[0]) == np.sign(xy[sh][0] - xy[other_sh][0])
        if outward and min(_angle(v, np.array([1.0, 0.0])), _angle(v, np.array([-1.0, 0.0]))) <= 25:
            return "out"
        return None

    a, b = arm(L_SH, L_EL, L_WR, R_SH), arm(R_SH, R_EL, R_WR, L_SH)
    if {a, b} != {"up", "out"}:
        return False
    return standing(kpts)


def standing(kpts: np.ndarray) -> bool:
    xy, ok = kpts[:, :2], kpts[:, 2] >= KP_CONF
    if not ok[[L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK]].all():
        return False
    for hip, knee, ank in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        if not _straight(xy[hip], xy[knee], xy[ank], 150):
            return False
        if _angle(xy[ank] - xy[hip], np.array([0.0, 1.0])) > 25:  # leg points down
            return False
    return True


@dataclass
class Event:
    t: float
    frame: int          # index into Detections frames
    det: int            # global detection index
    box: np.ndarray


@dataclass
class _Candidate:
    box: np.ndarray
    start: float
    last: float
    hits: int
    total: int
    centre0: np.ndarray


def scan(dets) -> tuple[list[Event], int]:
    """Find L-pose registrations. Returns (events, number of single frames showing the pose)."""
    events: list[Event] = []
    cands: list[_Candidate] = []
    cooldown_until = -1.0
    pose_frames = 0
    for i in range(len(dets)):
        t = float(dets.t[i])
        sl = dets.frame(i)
        boxes = dets.boxes[sl]
        flags = [l_pose(k) for k in dets.kpts[sl]]
        pose_frames += sum(flags)
        cands = [c for c in cands if t - c.last <= 0.5]
        for j, (b, f) in enumerate(zip(boxes, flags)):
            match = None
            if cands:
                ov = iou(b, np.array([c.box for c in cands]))
                if ov.max() > 0.3:
                    match = cands[int(np.argmax(ov))]
            if match is None:
                if not f:
                    continue
                c0 = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                cands.append(_Candidate(b, t, t, 1, 1, c0))
                continue
            match.box, match.last = b, t
            match.total += 1
            match.hits += int(f)
            h = b[3] - b[1]
            centre = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            if np.linalg.norm(centre - match.centre0) > MAX_DRIFT * h or match.hits / match.total < HOLD_FRACTION:
                cands.remove(match)  # moved or broke the pose: start over on the next pose frame
                continue
            if t - match.start >= HOLD_S and t >= cooldown_until:
                events.append(Event(t, i, sl.start + j, b.copy()))
                cooldown_until = t + COOLDOWN_S
                cands.remove(match)
    return events, pose_frames


def _person(arm_up: str = "left", standing_ok: bool = True, out_angle: float = 0.0,
            up_angle: float = 0.0) -> np.ndarray:
    """Synthetic COCO keypoints of a 170 px person facing the camera (for self-tests)."""
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = 0.9
    sh_l, sh_r = np.array([110.0, 100.0]), np.array([90.0, 100.0])  # image-left is person's right
    k[L_SH, :2], k[R_SH, :2] = sh_l, sh_r
    k[L_HIP, :2], k[R_HIP, :2] = [107, 160], [93, 160]
    if standing_ok:
        k[L_KNEE, :2], k[R_KNEE, :2] = [107, 205], [93, 205]
        k[L_ANK, :2], k[R_ANK, :2] = [107, 250], [93, 250]
    else:  # bent, spread legs (climbing)
        k[L_KNEE, :2], k[R_KNEE, :2] = [140, 175], [60, 175]
        k[L_ANK, :2], k[R_ANK, :2] = [140, 215], [60, 215]

    def put(sh, el, wr, base, ang_deg):
        r = np.radians(ang_deg)
        d = np.array([np.sin(r), -np.cos(r)])
        k[el, :2] = base + 25 * d
        k[wr, :2] = base + 50 * d

    if arm_up == "left":
        put(L_SH, L_EL, L_WR, sh_l, up_angle)                  # up
        put(R_SH, R_EL, R_WR, sh_r, -90 + out_angle)           # out to image-left
    else:
        put(R_SH, R_EL, R_WR, sh_r, -up_angle)
        put(L_SH, L_EL, L_WR, sh_l, 90 - out_angle)
    return k


def selftest() -> None:
    assert l_pose(_person("left")) and l_pose(_person("right"))
    assert l_pose(_person("left", out_angle=15, up_angle=20))
    assert not l_pose(_person("left", out_angle=40)), "out arm too far from horizontal"
    assert not l_pose(_person("left", up_angle=45)), "up arm too far from vertical"
    assert not l_pose(_person("left", standing_ok=False)), "climbing legs"
    y = _person("left")
    y[R_EL, :2], y[R_WR, :2] = y[R_SH, :2] + [-15, -20], y[R_SH, :2] + [-30, -40]  # Y: both up-ish
    assert not l_pose(y), "Y pose"

    # Hold logic on a 3 s, 10 fps sequence with a little jitter, plus a climber-like bystander.
    held = _Sequence(set(range(5, 26)))                         # holds 2.0 s
    events, _ = scan(held)
    assert len(events) == 1 and 2.0 <= events[0].t <= 2.1, [e.t for e in events]
    assert scan(_Sequence(set(range(5, 11)) | set(range(16, 22))))[0] == [], "broken hold fired"
    print("registration selftest ok")


class _Sequence:
    """Minimal Detections stand-in for the self-test: one posing person and one bystander."""

    def __init__(self, pose_frames: set[int], n: int = 30, seed: int = 0):
        rng = np.random.default_rng(seed)
        t, offsets, boxes, kpts = [], [0], [], []
        for f in range(n):
            k = _person("left") if f in pose_frames else _person("left", out_angle=60, up_angle=60)
            k[:, :2] += rng.normal(0, 1.0, (17, 2))
            other = _person("right", standing_ok=False)
            other[:, 0] += 300
            for kk in (k, other):
                boxes.append(np.r_[kk[:, :2].min(0) - 5, kk[:, :2].max(0) + 5])
                kpts.append(kk)
            t.append(f / 10)
            offsets.append(len(boxes))
        self.t, self.boxes, self.kpts, self.offsets = np.array(t), np.array(boxes), np.array(kpts), offsets

    def __len__(self) -> int:
        return len(self.t)

    def frame(self, i: int) -> slice:
        return slice(self.offsets[i], self.offsets[i + 1])


if __name__ == "__main__":
    selftest()
