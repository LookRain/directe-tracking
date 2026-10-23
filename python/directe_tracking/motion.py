"""Is the wall moving here? Confirms second-chance detections (deep dive:
docs/review/deep-dives/missed-climbers.md).

Fixed camera: a causal background model on a 270x480 copy, log-ratio illumination-normalised (so
IMG_3835's cycling lights cancel), sigma-delta approximate median, not learned under the predicted
climber (so a resting climber isn't absorbed). Moving camera: difference with the frame 1 s
earlier, registered onto this one (phone: gyro homography; lab: ORB + RANSAC).
Motion boxes are never candidates on their own: blobs are partial and lite re-ID can't judge them.
"""

from __future__ import annotations

import cv2
import numpy as np

from .tracker import iou

SW, SH = 270, 480  # working size (portrait)


class Background:
    """Sigma-delta approximate median of illumination-normalised frames: O(1) memory, phone-cheap."""

    def __init__(self, sigma: float = 25.0, delta: float = 0.004):
        self.sigma, self.delta = sigma, delta
        self.reset()

    def reset(self) -> None:
        self.bg = None
        self.n_upd = 0

    def norm(self, small: np.ndarray) -> np.ndarray:
        f = np.log(small.astype(np.float32) + 8.0)
        return f - cv2.GaussianBlur(f, (0, 0), self.sigma)

    def update(self, n: np.ndarray, freeze: np.ndarray | None) -> None:
        if self.bg is None:
            self.bg, self.n_upd = n.copy(), 1
            return
        step = self.delta * np.sign(n - self.bg)
        if self.n_upd < 50:  # warm-up: faster
            step *= 8
        if freeze is not None:
            step[freeze] = 0
        self.bg += step
        self.n_upd += 1

    def ready(self) -> bool:
        return self.bg is not None and self.n_upd >= 50


def blobs(d: np.ndarray, h_exp: float, k: float = 6.0, thr_abs: float = 0.25) -> list[np.ndarray]:
    """Foreground blobs at body-height scale (working px); h_exp = expected body height."""
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) * 1.4826
    thr = max(thr_abs, med + k * mad)
    fg = cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    min_area = max(4, int(0.004 * h_exp * h_exp))
    boxes = [[x, y, x + w, y + h] for x, y, w, h, a in stats[1:] if a >= min_area]
    # merge components within 0.35 body heights (a climber's moving limbs come apart)
    pad, merged = 0.35 * h_exp, True
    while merged and len(boxes) > 1:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] - pad <= b[2] and b[0] - pad <= a[2] and a[1] - pad <= b[3] and b[1] - pad <= a[3]:
                    nb = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    if nb[3] - nb[1] <= 1.6 * h_exp and nb[2] - nb[0] <= 1.6 * h_exp:
                        boxes[i] = nb
                        del boxes[j]
                        merged = True
                        break
            if merged:
                break
    out = []
    for b in boxes:
        w, h = b[2] - b[0], b[3] - b[1]
        if h > 2.5 * h_exp or w > 2.5 * h_exp or max(w, h) < 0.15 * h_exp:
            continue  # too big, or a speck
        if h > 4 * w and w < 0.12 * h_exp:
            continue  # rope-like sliver
        out.append(np.array(b, np.float32))
    return out


def register(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Homography mapping grey image a onto b (ORB + RANSAC), the gyro's stand-in. None if weak."""
    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 30:
        return None
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    H, _ = cv2.findHomography(pa, pb, cv2.RANSAC, 2.0)
    return H


class Evidence:
    """Motion blobs at ~5 Hz, in full-frame pixels; `supports(box)` asks whether something moved there."""

    def __init__(self, moving_camera: bool, hz: float = 5.0):
        self.moving, self.every = moving_camera, 1.0 / hz
        self.bg = Background()
        self.t, self.blob_boxes, self.hist = -1e9, [], []

    def update(self, t: float, frame: np.ndarray, pred_box: np.ndarray, reset: bool = False) -> None:
        if reset:
            self.bg.reset()
            self.hist.clear()
            self.blob_boxes = []
        if t - self.t < self.every:
            return
        self.t = t
        sx, sy = SW / frame.shape[1], SH / frame.shape[0]
        small = cv2.resize(frame, (SW, SH), interpolation=cv2.INTER_AREA)
        n = self.bg.norm(small)
        h = max(float(pred_box[3] - pred_box[1]) * sy, 8.0)
        if self.moving:
            d = self._registered_diff(t, small, n)
        else:
            d = np.abs(n - self.bg.bg).max(axis=2) if self.bg.ready() else None  # difference, then learn
            b, p = pred_box * [sx, sy, sx, sy], 0.5 * h
            freeze = np.zeros((SH, SW, 3), bool)  # don't learn the climber
            freeze[max(int(b[1] - p), 0):max(int(b[3] + p), 0), max(int(b[0] - p), 0):max(int(b[2] + p), 0)] = True
            self.bg.update(n, freeze)
        if d is None:
            self.blob_boxes = []
            return
        scale = np.array([sx, sy, sx, sy], np.float32)
        self.blob_boxes = [b / scale for b in blobs(cv2.GaussianBlur(d, (0, 0), 1.0), h)]

    def _registered_diff(self, t: float, small: np.ndarray, n: np.ndarray) -> np.ndarray | None:
        self.hist.append((t, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), n))
        while len(self.hist) > 2 and self.hist[1][0] <= t - 1.0:
            self.hist.pop(0)
        t0, old, n_old = self.hist[0]
        if t - t0 < 0.9:
            return None
        H = register(old, self.hist[-1][1])
        if H is None:
            return None
        warped = cv2.warpPerspective(n_old, H, (SW, SH), borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpPerspective(np.ones((SH, SW), np.uint8), H, (SW, SH))
        d = np.abs(n - warped).max(axis=2)
        d[cv2.erode(valid, np.ones((7, 7), np.uint8)) == 0] = 0
        return d

    def supports(self, box: np.ndarray) -> bool:
        for b in self.blob_boxes:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if iou(box, b[None])[0] >= 0.1 or (box[0] <= cx <= box[2] and box[1] <= cy <= box[3]):
                return True
        return False
