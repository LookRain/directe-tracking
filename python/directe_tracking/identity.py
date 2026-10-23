"""Identity features per detection: person segmentation + learned re-identification.

For every cached pose detection:
  - match a YOLO person-segmentation instance (on device, Apple Vision's person instance masks
    play this role) and keep colour sampling inside the body;
  - embed the person with OSNet-AIN (re-ID trained on MSMT17), both from the plain crop and from
    the crop with the background greyed out, so the two can be compared.

People the segmentation model finds but the pose model missed (common for a climber seen from
directly below near the top of a wall) are kept as extra detections: no keypoints, so no colour
regions, but a mask and a re-ID embedding.

Cached next to the detections as det_..._id_v{VERSION}.npz.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import appearance, detect, pipeline, video
from .clips import Clip, data_root

VERSION = 6
EMB_DIM = 512


def cache_path(clip: Clip) -> Path:
    det = detect.cache_path(clip)
    return det.with_name(det.stem + f"_id_v{VERSION}{pipeline.current().id_suffix}.npz")


class Segmenter:
    def __init__(self):
        from ultralytics import YOLO

        pipe = pipeline.current()
        self.model = YOLO(str(data_root() / "models" / pipe.seg))
        self.imgsz = pipe.seg_imgsz

    def __call__(self, frames: list[np.ndarray]) -> list[tuple[np.ndarray, list[np.ndarray]]]:
        """Per frame: (boxes[N,4], polygons) for people."""
        out = []
        for r in self.model.predict(frames, imgsz=self.imgsz, conf=0.2, classes=[0], device="mps",
                                    verbose=False, retina_masks=False):
            boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            polys = [p.astype(np.float32) for p in r.masks.xy] if r.masks is not None else []
            out.append((boxes, polys))
        return out


class Embedder:
    def __init__(self):
        from boxmot.reid.core.runtime import ReID

        self.reid = ReID(weights=data_root() / "models" / pipeline.current().reid, device="mps")

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, EMB_DIM), np.float32)
        return np.asarray(self.reid(crops), dtype=np.float32).reshape(len(crops), -1)


def masked_crop(frame: np.ndarray, box: np.ndarray, poly: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    H, W = frame.shape[:2]
    x0, y0 = max(int(box[0]), 0), max(int(box[1]), 0)
    x1, y1 = min(int(np.ceil(box[2])), W), min(int(np.ceil(box[3])), H)
    x1, y1 = max(x1, x0 + 2), max(y1, y0 + 2)
    crop = frame[y0:y1, x0:x1]
    if poly is None or len(poly) < 3:
        return crop, crop
    m = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(poly - [x0, y0]).astype(np.int32)], 1)
    grey = np.full_like(crop, 124)
    return crop, np.where(m[..., None].astype(bool), crop, grey)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-6)


def run(clip: Clip, batch: int = 8, force: bool = False) -> Path:
    out = cache_path(clip)
    if out.exists() and not force:
        print(f"[identity] cached {out}")
        return out
    dets = detect.Detections.load_raw(detect.cache_path(clip))
    seg, emb = Segmenter(), Embedder()
    N = len(dets.boxes)
    feat = np.zeros((N, len(appearance.REGIONS), appearance.DIM), np.float32)
    fvalid = np.zeros((N, len(appearance.REGIONS)), bool)
    E = np.zeros((N, EMB_DIM), np.float16)
    Em = np.zeros((N, EMB_DIM), np.float16)
    has_mask = np.zeros(N, bool)
    X_frame, X_box, X_emb, X_embm = [], [], [], []  # segmentation-only people
    t0 = time.time()
    pending: list[tuple[int, np.ndarray]] = []

    def flush():
        for (i, frame), (sboxes, spolys) in zip(pending, seg([p[1] for p in pending])):
            sl = dets.frame(i)
            boxes = dets.boxes[sl]
            iou = _iou_matrix(boxes, sboxes)
            crops, mcrops = [], []
            for j, k in enumerate(range(sl.start, sl.stop)):
                poly = None
                if iou.shape[1]:
                    m = int(np.argmax(iou[j]))
                    if iou[j, m] >= 0.5:
                        poly = spolys[m]
                has_mask[k] = poly is not None
                feat[k], fvalid[k] = appearance.features(frame, boxes[j], dets.kpts[k], poly)
                c, mc = masked_crop(frame, boxes[j], poly)
                crops.append(c); mcrops.append(mc)
            if crops:
                E[sl] = emb(crops)
                Em[sl] = emb(mcrops)
            unmatched = [m for m in range(len(sboxes)) if iou.shape[0] == 0 or iou[:, m].max() < 0.3]
            if unmatched:
                xc = [masked_crop(frame, sboxes[m], spolys[m] if m < len(spolys) else None) for m in unmatched]
                X_frame.extend([i] * len(unmatched))
                X_box.extend(sboxes[m] for m in unmatched)
                X_emb.append(emb([c for c, _ in xc]))
                X_embm.append(emb([mc for _, mc in xc]))
        pending.clear()

    for n, (i, (fi, frame)) in enumerate(zip(range(len(dets)), video.frames_at(clip.path, dets.fidx))):
        pending.append((i, frame))
        if len(pending) == batch:
            flush()
        if n % 400 == 0 and n:
            print(f"[identity] {clip.id} t={dets.t[i]:6.1f}s  {n / (time.time() - t0):4.1f} frames/s", flush=True)
    if pending:
        flush()
    xe = np.concatenate(X_emb) if X_emb else np.zeros((0, EMB_DIM), np.float32)
    xem = np.concatenate(X_embm) if X_embm else np.zeros((0, EMB_DIM), np.float32)
    np.savez_compressed(out, feat=feat, fvalid=fvalid, emb=E, emb_masked=Em, has_mask=has_mask,
                        x_frame=np.array(X_frame, np.int32), x_box=np.array(X_box, np.float32).reshape(-1, 4),
                        x_emb=xe.astype(np.float16), x_emb_masked=xem.astype(np.float16))
    print(f"[identity] wrote {out}: {N} detections, {has_mask.mean():.0%} with masks, "
          f"{len(X_frame)} segmentation-only people, {time.time() - t0:.0f} s")
    return out


def load_into(dets: detect.Detections, clip: Clip, extras: bool = True) -> bool:
    """Swap in masked colour features, attach embeddings and (optionally) merge the
    segmentation-only people into the per-frame detections, if the identity pass has run."""
    p = cache_path(clip)
    if not p.exists():
        return False
    z = np.load(p)
    feat, fvalid = z["feat"], z["fvalid"]
    emb, embm = z["emb"].astype(np.float32), z["emb_masked"].astype(np.float32)
    n = len(dets.boxes)
    source = np.zeros(n, np.int8)  # 0: pose detector, 1: segmentation only
    if extras and len(z["x_frame"]):
        xf, xb = z["x_frame"], z["x_box"]
        order = np.argsort(np.concatenate([np.repeat(np.arange(len(dets)), np.diff(dets.offsets)), xf]), kind="stable")
        cat = lambda a, b: np.concatenate([a, b])[order]  # noqa: E731
        m = len(xf)
        dets.boxes = cat(dets.boxes, xb)
        dets.conf = cat(dets.conf, np.zeros(m, np.float32))
        dets.kpts = cat(dets.kpts, np.zeros((m, 17, 3), np.float32))
        feat = cat(feat, np.zeros((m, *feat.shape[1:]), np.float32))
        fvalid = cat(fvalid, np.zeros((m, fvalid.shape[1]), bool))
        emb = cat(emb, z["x_emb"].astype(np.float32))
        embm = cat(embm, z["x_emb_masked"].astype(np.float32))
        source = cat(source, np.ones(m, np.int8))
        counts = np.diff(dets.offsets) + np.bincount(xf, minlength=len(dets))
        dets.offsets = np.concatenate([[0], np.cumsum(counts)])
    dets.feat, dets.fvalid, dets.emb, dets.emb_masked, dets.source = feat, fvalid, emb, embm, source
    return True
