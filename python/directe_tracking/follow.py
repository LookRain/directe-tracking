"""Follow the climber through detector misses by continuity: they were confirmed here a moment ago, something
the climber's size is still here and moving, so it's still them (deep dive: far climbers, the owner's idea).

A far, tiny or foreshortened climber drops in and out of the detector. Each tick the detector has nobody at the
prediction, this proposes one box where the climber went, from the last detector-confirmed box (the anchor):

  "ncc"   the anchor's patch, matched (normalised cross-correlation) in a window round the last position, at a
          working scale where the body is ~40 px. Accepted when the match is good and the matched spot moved
          within the last `alive_s` (a static hold that happens to correlate never moves), or the match is strong.
  "blob"  the frame difference (camera shift compensated) round the last position: the moving pixels connected to
          the last box, if they make a person-sized region; the box moves to their centroid.

Guards: at most `max_coast_s` since the last detector confirmation; the box keeps the anchor's size; a move of at
most `max_step` body heights per tick; nothing proposed on top of a person the detector did find (that person is
the tracker's call); after a camera bump (shift confidence low) nothing until the next confirmation.
The proposal is a candidate like any other: the tracker still judges its appearance and position, so identity stays
the tracker's call (the same pattern as online.second_chance).
"""

from __future__ import annotations

import cv2
import numpy as np

from .tracker import iou


class MotionFollow:
    def __init__(self, mode: str = "ncc", max_coast_s: float = 4.0, body_px: float = 40.0, ncc_min: float = 0.45,
                 ncc_still: float = 0.7, alive_s: float = 1.5, max_step: float = 0.6, search: float = 1.0):
        self.mode, self.max_coast_s, self.body_px = mode, max_coast_s, body_px
        self.ncc_min, self.ncc_still, self.alive_s = ncc_min, ncc_still, alive_s
        self.max_step, self.search = max_step, search
        self.anchor = None          # (t, box, template at working scale, scale)
        self.box = None             # the last followed box (the anchor's, then each proposal's)
        self.t_moved = -1e9         # last time the followed box contained motion
        self.prev = None            # previous grey frame
        self.proposed = None        # this tick's proposal (box) or None
        self.stats = {"proposed": 0, "accepted": 0, "ticks": 0}

    # ------------------------------------------------------------------------------------------- per tick
    def propose(self, t: float, frame: np.ndarray, shift: np.ndarray, shift_conf: float, pred: np.ndarray,
                boxes: np.ndarray) -> np.ndarray | None:
        """Called before the tracker's step with this tick's detections; returns a box or None."""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev, self.prev = self.prev, g
        self.proposed = None
        self.stats["ticks"] += 1
        shift = np.asarray(shift, np.float64)
        if self.box is not None:
            self.box = self.box + np.tile(shift, 2)   # the camera moved the image
        if shift_conf < 0.1:   # bumped: the background can't be compared, and the climber may be anywhere
            self.anchor = None
            return None
        if self.anchor is None or self.box is None or prev is None or t - self.anchor[0] > self.max_coast_s:
            return None
        if len(boxes) and (iou(pred, boxes) >= 0.3).any():
            return None   # the detector has someone at the prediction: nothing to do
        H, W = g.shape
        b = self.box
        h = float(b[3] - b[1])
        w = float(b[2] - b[0])
        r = self.search * h + 2.0
        x0, y0 = int(max(b[0] - r, 0)), int(max(b[1] - r, 0))
        x1, y1 = int(min(b[2] + r, W)), int(min(b[3] + r, H))
        if x1 - x0 < w + 2 or y1 - y0 < h + 2:
            return None
        # motion: this frame minus the last, the last moved by the camera shift
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        pw = cv2.warpAffine(prev, M, (W, H), borderMode=cv2.BORDER_REPLICATE) if np.hypot(*shift) > 0.3 else prev
        d = cv2.absdiff(g[y0:y1, x0:x1], pw[y0:y1, x0:x1]).astype(np.float32)
        d = cv2.GaussianBlur(d, (0, 0), max(0.6, 0.02 * h))
        med = float(np.median(d))
        thr = max(6.0, med + 5.0 * 1.4826 * float(np.median(np.abs(d - med))))
        moving = d > thr
        if self.mode == "blob":
            nb = self._blob(moving, b, x0, y0, h, w)
        else:
            nb = self._ncc(g, x0, y0, x1, y1, h, w, moving, t)
        if nb is None:
            return None
        c_old, c_new = (b[:2] + b[2:]) / 2, (nb[:2] + nb[2:]) / 2
        if np.linalg.norm(c_new - c_old) > self.max_step * h:
            return None
        if len(boxes) and (iou(nb, boxes) >= 0.3).any():
            return None   # that's someone the detector sees: the tracker decides about them
        self.box = nb
        self.proposed = nb
        self.stats["proposed"] += 1
        return nb

    def after(self, t: float, frame: np.ndarray, state: str, box: np.ndarray, chosen_box: np.ndarray | None) -> None:
        """After the tracker's step: a lock on a detector box re-anchors; a lock on our proposal moves the follow."""
        if state != "locked" or chosen_box is None:
            if state == "lost":
                self.box = None
            return
        if self.proposed is not None and iou(chosen_box, self.proposed[None])[0] > 0.9:
            self.stats["accepted"] += 1
            return   # coasting on our own box: the anchor stays the last detector confirmation
        h = float(chosen_box[3] - chosen_box[1])
        if h < 6:
            return
        s = float(np.clip(self.body_px / h, 0.35, 3.0))
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x0, y0, x1, y1 = (int(round(v)) for v in chosen_box)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, g.shape[1]), min(y1, g.shape[0])
        if x1 - x0 < 3 or y1 - y0 < 3:
            return
        tpl = cv2.resize(g[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        self.anchor = (t, chosen_box.astype(np.float64).copy(), tpl, s)
        self.box = chosen_box.astype(np.float64).copy()
        self.t_moved = t

    # ---------------------------------------------------------------------------------------- the two modes
    def _ncc(self, g, x0, y0, x1, y1, h, w, moving, t):
        _, abox, tpl, s = self.anchor
        roi = cv2.resize(g[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if roi.shape[0] <= tpl.shape[0] or roi.shape[1] <= tpl.shape[1] or tpl.std() < 3:
            return None
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, v, _, (px, py) = cv2.minMaxLoc(res)
        if v < self.ncc_min:
            return None
        bw, bh = abox[2] - abox[0], abox[3] - abox[1]
        nb = np.array([x0 + px / s, y0 + py / s, x0 + px / s + bw, y0 + py / s + bh])
        # alive: something moved inside the matched box (shrunk to its middle, away from edges of holds)
        mx0, my0 = int(nb[0] - x0 + 0.15 * bw), int(nb[1] - y0 + 0.1 * bh)
        mx1, my1 = int(nb[2] - x0 - 0.15 * bw), int(nb[3] - y0 - 0.1 * bh)
        sub = moving[max(my0, 0):max(my1, 1), max(mx0, 0):max(mx1, 1)]
        if sub.size and sub.mean() > 0.03:
            self.t_moved = t
        if v >= self.ncc_still or t - self.t_moved <= self.alive_s:
            return nb
        return None

    def _blob(self, moving, b, x0, y0, h, w):
        n, lab, stats, cent = cv2.connectedComponentsWithStats(moving.astype(np.uint8), connectivity=8)
        if n <= 1:
            return b.copy()   # nothing moved: still there (bounded by max_coast_s)
        lb = np.array([b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0])
        pad = 0.3 * h
        keep = []
        for k in range(1, n):
            x, y, ww, hh, a = stats[k]
            if a < max(3, 0.002 * h * h):
                continue
            if x - pad <= lb[2] and lb[0] <= x + ww + pad and y - pad <= lb[3] and lb[1] <= y + hh + pad:
                keep.append(k)
        if not keep:
            return b.copy()
        sel = np.isin(lab, keep)
        ys, xs = np.nonzero(sel)
        ext_w, ext_h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if ext_h > 2.0 * h or ext_w > 2.0 * max(w, 0.5 * h):
            return None   # bigger than a person: lights, a crowd, a swinging rope across the window
        cx, cy = x0 + xs.mean(), y0 + ys.mean()
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
