"""Per-region colour signatures of a person, sampled only inside the body.

Regions come only from confident pose keypoints (COCO-17). There is deliberately no fallback to
fixed slices of the box: on a climbing wall those slices are mostly wall and holds, and a white
wall with orange holds "matches" a climber in a white tee and orange shorts. A region the
keypoints can't place is unknown, not a match. Each region gets a hue histogram of the coloured pixels plus five brightness bins for
the grey/black/white ones, so a white harness and a white chalk bag still count as a signal.
"""

from __future__ import annotations

import cv2
import numpy as np

VERSION = 4  # bump when features change; cached features are recomputed

REGIONS = ("head", "torso", "hips", "legs")
WEIGHTS = np.array([0.15, 0.35, 0.2, 0.3], dtype=np.float32)
HUE_BINS = 12
GREY_EDGES = (45, 90, 150, 200)  # V thresholds for the non-coloured pixels: 5 brightness levels
DIM = HUE_BINS + len(GREY_EDGES) + 1
KP_CONF = 0.35
MIN_PIXELS = 15

# COCO keypoint indices
L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE = 5, 6, 11, 12, 13, 14
HEAD = (0, 1, 2, 3, 4)

def _quad(p: np.ndarray, q: np.ndarray, half_width: float) -> np.ndarray:
    d = q - p
    n = np.array([-d[1], d[0]], dtype=np.float32)
    n /= max(np.linalg.norm(n), 1e-6)
    n *= half_width
    return np.array([p + n, q + n, q - n, p - n], dtype=np.float32)


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def region_polygons(box: np.ndarray, kpts: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Body regions placed by keypoints; regions that can't be placed are left out."""
    ok = kpts[:, 2] >= KP_CONF
    xy = kpts[:, :2].astype(np.float32)
    polys: dict[str, list[np.ndarray]] = {}
    box_area = max(float((box[2] - box[0]) * (box[3] - box[1])), 1.0)

    if ok[[L_SH, R_SH, L_HIP, R_HIP]].all():
        sm = (xy[L_SH] + xy[R_SH]) / 2
        hm = (xy[L_HIP] + xy[R_HIP]) / 2
        up = sm - hm
        length = float(np.linalg.norm(up))
        hip_vec = xy[R_HIP] - xy[L_HIP]
        hip_w = max(float(np.linalg.norm(hip_vec)), 0.25 * length, 4.0)
        torso = np.array([xy[L_SH], xy[R_SH], xy[R_HIP], xy[L_HIP]])
        # A torso folded flat (bent over, seen from below) is a sliver: unreliable, skip it.
        if length > 6 and _area(torso) >= 0.03 * box_area:
            c = torso.mean(axis=0)
            polys["torso"] = [c + 0.75 * (torso - c)]
            along = hip_vec / max(np.linalg.norm(hip_vec), 1e-6) * (0.65 * hip_w)
            across = up * 0.15
            polys["hips"] = [np.array([hm - along + across, hm + along + across,
                                       hm + along - across, hm - along - across])]
        if ok[list(HEAD)].any() and length > 6:
            hc = xy[list(HEAD)][ok[list(HEAD)]].mean(axis=0)
            r = 0.15 * length
            ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            polys["head"] = [np.stack([hc[0] + r * np.cos(ang), hc[1] + r * np.sin(ang)], 1)]
        legs = []  # top of the thigh only: the shorts/trousers, not bare skin (everyone has skin)
        for hip, knee in ((L_HIP, L_KNEE), (R_HIP, R_KNEE)):
            if ok[knee] and np.linalg.norm(xy[knee] - xy[hip]) > 4:
                mid = xy[hip] + 0.45 * (xy[knee] - xy[hip])
                legs.append(_quad(xy[hip], mid, max(0.15 * hip_w, 2.0)))
        if legs:
            polys["legs"] = legs
    return polys


def _hist(hsv_pixels: np.ndarray) -> np.ndarray:
    h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
    chroma = (s >= 50) & (v >= 40)
    out = np.zeros(DIM, dtype=np.float32)
    if chroma.any():
        bins = (h[chroma].astype(np.int32) * HUE_BINS) // 180
        hue = np.bincount(bins, minlength=HUE_BINS).astype(np.float32)
        out[:HUE_BINS] = 0.25 * np.roll(hue, 1) + 0.5 * hue + 0.25 * np.roll(hue, -1)
    grey = np.bincount(np.searchsorted(GREY_EDGES, v[~chroma], side="right"),
                       minlength=len(GREY_EDGES) + 1).astype(np.float32)
    out[HUE_BINS:] = 0.15 * np.r_[grey[1:], 0] + 0.7 * grey + 0.15 * np.r_[0, grey[:-1]]
    return out / max(out.sum(), 1.0)


def features(frame: np.ndarray, box: np.ndarray, kpts: np.ndarray,
             body: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (feat[4, DIM], valid[4]) for one person.

    `body` is an optional person-segmentation polygon (frame coordinates); when given, only
    pixels inside it count, so a region that strays onto the wall loses those pixels.
    """
    H, W = frame.shape[:2]
    x0, y0 = max(int(box[0]) - 2, 0), max(int(box[1]) - 2, 0)
    x1, y1 = min(int(box[2]) + 3, W), min(int(box[3]) + 3, H)
    feat = np.zeros((len(REGIONS), DIM), dtype=np.float32)
    valid = np.zeros(len(REGIONS), dtype=bool)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return feat, valid
    crop = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    offset = np.array([x0, y0], dtype=np.float32)
    body_mask = None
    if body is not None and len(body) >= 3:
        body_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(body_mask, [np.round(body - offset).astype(np.int32)], 1)
    polys = region_polygons(box, kpts)
    for i, name in enumerate(REGIONS):
        if name not in polys:
            continue
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        for poly in polys[name]:
            cv2.fillConvexPoly(mask, np.round(poly - offset).astype(np.int32), 1)
        if body_mask is not None:
            mask &= body_mask
        px = hsv[mask.astype(bool)]
        if len(px) >= MIN_PIXELS:
            feat[i] = _hist(px)
            valid[i] = True
    return feat, valid


def similarity(fa: np.ndarray, va: np.ndarray, fb: np.ndarray, vb: np.ndarray) -> float:
    """Weighted Bhattacharyya coefficient over regions valid in both, or UNKNOWN."""
    return float(similarity_many(fa, va, fb[None], vb[None])[0])


UNKNOWN = -1.0
MIN_EVIDENCE = 0.3   # total region weight needed before a similarity means anything
VETO_BELOW = 0.55    # a clearly wrong region (black shorts vs orange) isn't averaged away
VETO_GAIN = 0.8


def similarity_many(profile: np.ndarray, pvalid: np.ndarray, feats: np.ndarray, valids: np.ndarray) -> np.ndarray:
    """Similarity per candidate in [0, 1], or UNKNOWN when too little of the body was seen."""
    both = valids & pvalid[None]
    bc = np.sqrt(feats * profile[None]).sum(axis=2)
    w = WEIGHTS[None] * both
    denom = w.sum(axis=1)
    mean = (bc * w).sum(axis=1) / np.maximum(denom, 1e-6)
    worst = np.where(both & (WEIGHTS[None] >= 0.2), bc, 1.0).min(axis=1)  # torso, hips, legs
    sim = np.clip(mean - VETO_GAIN * np.maximum(0.0, VETO_BELOW - worst), 0.0, 1.0)
    return np.where(denom >= MIN_EVIDENCE, sim, UNKNOWN)
