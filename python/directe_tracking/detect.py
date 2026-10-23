"""Person + pose detection with appearance features, cached per clip on the data drive."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import appearance, pipeline, video
from .clips import Clip, data_root

DEFAULT_MODEL = "yolo26m-pose.pt"  # the "full" pipeline; see lab/pipeline.py


class Detector:
    """YOLO pose on whole frames. Weights download into $DIRECTE_DATA/models."""

    def __init__(self, model: str | None = None, imgsz: int | None = None, conf: float = 0.25):
        from ultralytics import YOLO

        pipe = pipeline.current()
        model, imgsz = model or pipe.pose, imgsz or pipe.pose_imgsz

        self.name = model
        self.imgsz = imgsz
        self.conf = conf
        self.model = YOLO(str(data_root() / "models" / model))

    def __call__(self, frames: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per frame: boxes[N,4] xyxy, conf[N], kpts[N,17,3]."""
        results = self.model.predict(frames, imgsz=self.imgsz, conf=self.conf, classes=[0],
                                     device="mps", verbose=False)
        out = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = r.boxes.conf.cpu().numpy().astype(np.float32)
            if r.keypoints is not None and len(boxes):
                kpts = r.keypoints.data.cpu().numpy().astype(np.float32)
            else:
                kpts = np.zeros((len(boxes), 17, 3), dtype=np.float32)
            out.append((boxes, conf, kpts))
        return out


@dataclass
class Detections:
    """Flat arrays; detections of frame i are rows offsets[i]:offsets[i+1]."""

    t: np.ndarray          # [F] seconds
    fidx: np.ndarray       # [F] source frame index
    offsets: np.ndarray    # [F+1]
    boxes: np.ndarray      # [N,4] xyxy, full-frame pixels
    conf: np.ndarray       # [N]
    kpts: np.ndarray       # [N,17,3]
    feat: np.ndarray       # [N,4,DIM]
    fvalid: np.ndarray     # [N,4]
    size: tuple[int, int]  # (W, H)
    meta: dict

    def frame(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def __len__(self) -> int:
        return len(self.t)

    @classmethod
    def load_raw(cls, path: Path) -> "Detections":
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        n = len(z["boxes"])
        empty_f = np.zeros((n, len(appearance.REGIONS), appearance.DIM), np.float32)
        return cls(z["t"], z["fidx"], z["offsets"], z["boxes"], z["conf"], z["kpts"], empty_f,
                   np.zeros((n, len(appearance.REGIONS)), bool), tuple(meta["size"]), meta)


def cache_path(clip: Clip, fps: float = 10, model: str | None = None, imgsz: int | None = None) -> Path:
    pipe = pipeline.current()
    model, imgsz = model or pipe.pose, imgsz or pipe.pose_imgsz
    return clip.run_dir / f"det_{Path(model).stem}_{imgsz}_{fps:g}fps.npz"


def load(clip: Clip, fps: float = 10, model: str | None = None, imgsz: int | None = None,
         identity: bool = True) -> Detections:
    """Cached detections with appearance features. With `identity`, uses the segmentation-masked
    colour features and re-ID embeddings from `lab identity` when that pass has run; otherwise
    keypoint-only colour features, (re)computed when the appearance version changes."""
    det_path = cache_path(clip, fps, model, imgsz)
    dets = Detections.load_raw(det_path)
    if identity:
        from . import identity as ident
        if ident.load_into(dets, clip):
            return dets
    feat_path = det_path.with_name(det_path.stem + f"_feat_v{appearance.VERSION}.npz")
    if not feat_path.exists():
        t0 = time.time()
        feat, fvalid = dets.feat, dets.fvalid
        for i, (fi, frame) in enumerate(video.frames_at(clip.path, dets.fidx)):
            sl = dets.frame(i)
            for k in range(sl.start, sl.stop):
                feat[k], fvalid[k] = appearance.features(frame, dets.boxes[k], dets.kpts[k])
        np.savez_compressed(feat_path, feat=feat, fvalid=fvalid)
        print(f"[features] {clip.id} v{appearance.VERSION}: {len(dets.boxes)} detections in {time.time() - t0:.1f} s")
    z = np.load(feat_path)
    dets.feat, dets.fvalid = z["feat"], z["fvalid"]
    return dets


def run(clip: Clip, fps: float = 10, model: str | None = None, imgsz: int | None = None,
        batch: int = 8, force: bool = False) -> Path:
    pipe = pipeline.current()
    model, imgsz = model or pipe.pose, imgsz or pipe.pose_imgsz
    out = cache_path(clip, fps, model, imgsz)
    if out.exists() and not force:
        print(f"[detect] cached {out}")
        return out
    det = Detector(model, imgsz)
    vinfo = video.info(clip.path)
    T, FI, OFF, B, C, K = [], [], [0], [], [], []
    t0 = time.time()
    pending: list[tuple[int, float, np.ndarray]] = []

    def flush():
        for (fi, t, frame), (boxes, conf, kpts) in zip(pending, det([p[2] for p in pending])):
            T.append(t)
            FI.append(fi)
            B.extend(boxes); C.extend(conf); K.extend(kpts)
            OFF.append(len(B))
        pending.clear()

    for item in video.sample(clip.path, fps):
        pending.append(item)
        if len(pending) == batch:
            flush()
            n = len(T)
            if n % 200 < batch:
                el = time.time() - t0
                print(f"[detect] {clip.id} t={T[-1]:6.1f}/{vinfo['duration']:.0f}s  "
                      f"{n / el:5.1f} frames/s  dets={len(B)}", flush=True)
    if pending:
        flush()

    meta = {"clip": clip.id, "fps": fps, "model": model, "imgsz": imgsz,
            "size": [vinfo["width"], vinfo["height"]], "seconds": round(time.time() - t0, 1)}
    np.savez_compressed(
        out, t=np.array(T, np.float32), fidx=np.array(FI, np.int32), offsets=np.array(OFF, np.int64),
        boxes=np.array(B, np.float32).reshape(-1, 4), conf=np.array(C, np.float32),
        kpts=np.array(K, np.float32).reshape(-1, 17, 3), meta=json.dumps(meta))
    print(f"[detect] wrote {out} ({len(T)} frames, {len(B)} detections, {meta['seconds']} s)")
    return out
