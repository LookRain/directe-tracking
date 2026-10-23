"""Online "phone" runner: detection inside the tracking loop, the way the app will do it.

Each tick (10 Hz):
  1. focus pass: crop a window around where the tracker predicts the climber, at native
     resolution (640x640 px, or larger when the climber is big), and run the small pose and
     mask models on it, so a 60 px climber stays 60 px instead of shrinking to 20;
  2. wide pass: the small pose model on the whole frame at 640 px, to find big people and to
     re-find the climber after a loss;
  3. merge (focus detections win on overlap), describe (colour inside masks, re-ID embedding),
     and step the same TargetTracker the offline lab uses.

Works with any pipeline (`--pipeline lite` is the point). Results are cached per clip and scored
against labels/<clip>.json and the hard-segment benchmark samples.
"""

from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from . import appearance, identity, labels, motion, pipeline, video
from .clips import Clip
from .detect import Detector
from .tracker import LOCKED, Cands, Config, TargetTracker, iou, same_person, tag

FOCUS = 640          # native-resolution focus window side (px), minimum
FOCUS_K = 3.0        # window side = max(FOCUS, FOCUS_K x climber height)
# Second chance: when nobody is at the prediction, look again with the climber at ~400 px in the
# network input (the focus pass caps them at ~213). Lite misses: 49/58 boxed vs 18/58
# (docs/review/deep-dives/missed-climbers.md).
SC_K = 1.6            # window side in predicted body heights
SC_MIN = 213          # px; at most 3x upsampling
SC_REACH = 1.0        # accept a box centred within this many predicted body heights...
SC_SIZE = (0.5, 2.0)  # ...and this tall relative to the prediction


class Models:
    def __init__(self):
        pipe = pipeline.current()
        self.pose = Detector(pipe.pose, pipe.pose_imgsz)
        self.seg = identity.Segmenter()
        self.emb = identity.Embedder()
        self.calls = {"pose": 0, "seg": 0, "reid": 0, "second": 0}
        self.plain_reid = False  # embed plain crops: the lite masks' grey fill makes everyone look alike


class CameraMotion:
    """Whole-image shift between ticks (phase correlation on a small greyscale frame, ~1 ms): the
    lab's stand-in for the phone's gyro, which measures the same thing exactly and for free."""

    W = 360

    def __init__(self):
        self.prev = self.win = None

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        s = frame.shape[1] / self.W
        g = cv2.cvtColor(cv2.resize(frame, (self.W, int(round(frame.shape[0] / s)))), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.win is None or self.win.shape != g.shape:
            self.win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        prev, self.prev = self.prev, g
        if prev is None or prev.shape != g.shape:
            return np.zeros(2), 1.0
        (dx, dy), r = cv2.phaseCorrelate(prev, g, self.win)
        return np.array([dx, dy]) * s, float(r)


def _window(cx: float, cy: float, side: float, W: int, H: int) -> tuple[int, int, int, int]:
    side = int(min(side, W, H))
    x0 = int(np.clip(cx - side / 2, 0, W - side))
    y0 = int(np.clip(cy - side / 2, 0, H - side))
    return x0, y0, x0 + side, y0 + side


def perceive(m: Models, frame: np.ndarray, focus: tuple[int, int, int, int] | None) -> Cands:
    """Detections from the wide pass and (optionally) a focus window, described for the tracker."""
    H, W = frame.shape[:2]
    boxes, kpts, polys = [], [], []
    wb, _, wk = m.pose([frame])[0]
    m.calls["pose"] += 1
    if focus is not None:
        x0, y0, x1, y1 = focus
        crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
        fb, _, fk = m.pose([crop])[0]
        (sb, sp), = m.seg([crop])
        m.calls["pose"] += 1
        m.calls["seg"] += 1
        off = np.array([x0, y0, x0, y0], np.float32)
        fb = fb + off
        fk = fk.copy()
        fk[:, :, :2] += [x0, y0]
        sb = sb + off
        sp = [p + [x0, y0] for p in sp]
        used = set()
        for b, k in zip(fb, fk):
            poly = None
            if len(sb):
                ov = iou(b, sb)
                j = int(np.argmax(ov))
                if ov[j] >= 0.5:
                    poly, _ = sp[j], used.add(j)
            boxes.append(b); kpts.append(k); polys.append(poly)
        # People only the mask model found. A mask of just the upper body sits inside the pose
        # box with IoU ~0.3, so test containment too (it once became a "second person").
        for j in range(len(sb)):
            if j not in used and (not len(fb) or not same_person(sb[j], fb, 0.3).any()):
                boxes.append(sb[j]); kpts.append(np.zeros((17, 3), np.float32)); polys.append(sp[j] if j < len(sp) else None)
        # wide-pass people not already seen in the focus window
        for b, k in zip(wb, wk):
            inside = b[0] >= x0 and b[1] >= y0 and b[2] <= x1 and b[3] <= y1
            if not inside and (not boxes or not same_person(b, np.array(boxes)).any()):
                boxes.append(b); kpts.append(k); polys.append(None)
    else:
        boxes, kpts, polys = list(wb), list(wk), [None] * len(wb)
    return _describe(m, frame, boxes, kpts, polys)


def _describe(m: Models, frame: np.ndarray, boxes, kpts, polys) -> Cands:
    """Colour inside masks and re-ID embeddings for these people."""
    n = len(boxes)
    feats = np.zeros((n, len(appearance.REGIONS), appearance.DIM), np.float32)
    valids = np.zeros((n, len(appearance.REGIONS)), bool)
    crops = []
    for i, (b, k, p) in enumerate(zip(boxes, kpts, polys)):
        feats[i], valids[i] = appearance.features(frame, b, k, p)
        plain, masked = identity.masked_crop(frame, b, p)
        crops.append(plain if m.plain_reid else masked)
    embs = m.emb(crops) if n else np.zeros((0, identity.EMB_DIM), np.float32)
    m.calls["reid"] += n
    return Cands(np.array(boxes, np.float32).reshape(-1, 4), feats, valids, embs)


def second_chance(m: Models, frame: np.ndarray, pb: np.ndarray, c: Cands, mot: motion.Evidence | None) -> Cands:
    """Nobody at the prediction: look again at ~2x (pose + masks) and add at most one person at the
    prediction, if the wall's motion supports it (static boxes on holds and rock are what the
    larger view hallucinates). Identity is still the tracker's call."""
    H, W = frame.shape[:2]
    h = float(pb[3] - pb[1])
    if h <= 0 or (len(c) and (iou(pb, c.boxes) >= 0.3).any()):
        return c
    x0, y0, x1, y1 = _window((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2, float(np.clip(SC_K * h, SC_MIN, min(W, H))), W, H)
    s = 640 / (y1 - y0)
    crop = cv2.resize(frame[y0:y1, x0:x1], (640, 640), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
    fb, _, fk = m.pose([crop])[0]
    (sb, sp), = m.seg([crop])
    m.calls["pose"] += 1
    m.calls["seg"] += 1
    m.calls["second"] += 1
    off = np.array([x0, y0, x0, y0], np.float32)
    fb, sb = fb / s + off, sb / s + off
    fk = fk.copy()
    fk[:, :, :2] = fk[:, :, :2] / s + [x0, y0]
    sp = [p / s + [x0, y0] for p in sp]
    boxes, kpts, polys, used = [], [], [], set()
    for b, k in zip(fb, fk):  # pose boxes with their mask, as in perceive()
        poly = None
        if len(sb):
            ov = iou(b, sb)
            j = int(np.argmax(ov))
            if ov[j] >= 0.5:
                poly = sp[j]
                used.add(j)
        boxes.append(b); kpts.append(k); polys.append(poly)
    for j in range(len(sb)):  # people only the mask model found
        if j not in used and (not len(fb) or not same_person(sb[j], fb, 0.3).any()):
            boxes.append(sb[j]); kpts.append(np.zeros((17, 3), np.float32)); polys.append(sp[j] if j < len(sp) else None)
    if not boxes:
        return c
    B = np.array(boxes, np.float32)
    d = np.linalg.norm((B[:, :2] + B[:, 2:]) / 2 - (pb[:2] + pb[2:]) / 2, axis=1) / h
    r = (B[:, 3] - B[:, 1]) / h
    ok = (d <= SC_REACH) & (r >= SC_SIZE[0]) & (r <= SC_SIZE[1])
    if mot is not None:
        ok &= np.array([mot.supports(b) for b in B], bool)
    if len(c):
        ok &= np.array([not same_person(b, c.boxes).any() for b in B], bool)
    if not ok.any():
        return c
    i = int(np.argmin(np.where(ok, d, np.inf)))
    return c.concat(_describe(m, frame, [B[i]], [kpts[i]], [polys[i]]))


def run(clip: Clip, cfg: Config | None = None, fps: float = 10, end: float | None = None, referee=None) -> dict:
    pipe = pipeline.current()
    out = clip.run_dir / f"online_{pipe.name}{_suffix(referee)}.npz"
    m = Models()
    cfg = cfg or Config()
    opts = dict(VARIANTS.get(os.environ.get("DIRECTE_VARIANT", ""), {}))
    run_opts = {k: opts.pop(k) for k in list(opts) if k in RUNNER_KEYS}
    if opts:
        from dataclasses import replace
        cfg = replace(cfg, **opts)
    if clip.handheld and not run_opts.get("ground_handheld"):
        # (with ground_handheld, trk.camera() moves the ground line with the wall instead)
        from dataclasses import replace
        cfg = replace(cfg, ground_rule=False)

    t_seed = clip.seed[0]
    frames = video.sample(clip.path, fps, start=np.floor(t_seed * fps) / fps, end=end)
    T, FI, STATE, BOX, FLAGS = [], [], [], [], []  # FLAGS: jump refused, camera moving
    trk = None
    t0 = time.time()
    tag_window: list[tuple[float, Cands, int]] = []
    W = H = None
    cam = CameraMotion()
    mot = motion.Evidence(moving_camera=clip.handheld) if run_opts.get("motion_confirm") else None
    m.plain_reid = bool(run_opts.get("plain_reid"))
    for fi, t, frame in frames:
        H, W = frame.shape[:2]
        shift, cam_conf = cam(frame)
        if trk is None:
            # Tag moment: look around the tap point at full resolution, then build the profile
            # from the next 1.5 s exactly as the offline lab does.
            px, py = clip.seed[1] * W, clip.seed[2] * H
            c = perceive(m, frame, _window(px, py, FOCUS * 1.5, W, H))
            tag_window.append((t, c, fi, frame))
            if t - tag_window[0][0] < 1.5:
                continue
            trk, box0, k0 = _start(tag_window, px, py, W, H, cfg)
            if referee is not None:
                referee.clip_id = clip.id + "/online"
                referee.set_tag(tag_window[k0][3], box0, tag_window[k0][0])
            prev = None
            for tt, cc, ff, _ in tag_window:
                s = trk.step(tt, cc)
                T.append(tt); FI.append(ff); STATE.append(s.state); BOX.append(s.box); FLAGS.append((False, False))
                prev = s
            continue
        trk.camera(shift, cam_conf)
        pb = trk.predicted_box(t)
        if mot is not None:  # a knocked fixed phone invalidates the background
            bump = not clip.handheld and (np.hypot(*shift) > 6 * W / 1080 or cam_conf < 0.1)
            mot.update(t, frame, pb, reset=bump)
        side = max(FOCUS, FOCUS_K * (pb[3] - pb[1]))
        # While lost, keep looking at full resolution (with masks) where the climber was last seen:
        # the wide pass alone sees that spot 3-6x downscaled and without masks (outside review F4).
        if trk.state == "lost" and not run_opts.get("lost_focus"):
            focus = None
        else:
            focus = _window((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2, side, W, H)
        if referee is not None and trk.state != LOCKED:
            hint = referee.search_window(t, W, H)  # purpose 2: search where the referee says
            if hint is not None:
                focus = hint
        c = perceive(m, frame, focus)
        if run_opts.get("second_chance"):
            c = second_chance(m, frame, pb, c, mot)
        if referee is not None:
            trk.exclude = referee.tick(t, c.boxes, trk, prev, lambda frame=frame: frame, trk.rival_gap)
        s = trk.step(t, c)
        if referee is not None and s.state == LOCKED and s.app >= 0.85 and trk.steady():
            referee.note_confident(lambda frame=frame: frame, s.box, t)
        prev = s
        T.append(t); FI.append(fi); STATE.append(s.state); BOX.append(s.box)
        FLAGS.append((trk.jump_t == t, trk.unstable))
        if len(T) % 300 == 0:
            print(f"[online] {clip.id} t={t:6.1f}s  {len(T) / (time.time() - t0):4.1f} ticks/s", flush=True)
    np.savez_compressed(out, t=np.array(T), fidx=np.array(FI), state=np.array(STATE), box=np.array(BOX, np.float32),
                        jump=np.array([f[0] for f in FLAGS]), unstable=np.array([f[1] for f in FLAGS]),
                        meta=np.array(json.dumps(_stamp(cfg, run_opts, referee))))
    if referee is not None:
        out.with_suffix(".log.json").write_text(json.dumps(referee.log, indent=0, default=str))
    calls = {k: round(v / max(len(T), 1), 2) for k, v in m.calls.items()}
    print(f"[online] {clip.id} {pipe.name}: {len(T)} ticks in {time.time() - t0:.0f} s; calls per tick {calls}")
    return {"ticks": len(T), "calls_per_tick": calls}


class _Fake:
    """Just enough of Detections for tracker.tag()."""

    def __init__(self, window, W, H):
        self.t = np.array([w[0] for w in window])
        self.size = (W, H)
        self.offsets = np.cumsum([0] + [len(w[1]) for w in window])
        self.boxes = np.concatenate([w[1].boxes for w in window])
        self.feat = np.concatenate([w[1].feats for w in window])
        self.fvalid = np.concatenate([w[1].valids for w in window])
        self.emb_masked = np.concatenate([w[1].embs for w in window])

    def __len__(self):
        return len(self.t)

    def frame(self, i):
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))


def _start(window, px, py, W, H, cfg):
    fake = _Fake(window, W, H)
    profile, box, start = tag(fake, (window[0][0], px / W, py / H))
    return TargetTracker(profile, box, float(fake.t[start]), cfg, frame_h=H), box, start


# DIRECTE_VARIANT=<name>: try tracker/runner changes side by side (outputs get a _<name> suffix).
# Keys that are Config fields go to the tracker; "lost_focus" is the runner's.
VARIANTS = {
    "strict": {"others_always": True},
    "qw": {"learn_gate": True, "fall_size": True, "lost_focus": True},  # outside-review quick wins
    "lostfocus": {"lost_focus": True},
    "learngate": {"learn_gate": True},
    "fallsize": {"fall_size": True},
    "qw2": {"learn_gate": True, "fall_size": True, "lost_focus": True},  # repeat of qw (determinism check)
    "lgfs": {"learn_gate": True, "fall_size": True},
    "second": {"second_chance": True},  # deep dive: missed climbers
    "second_motion": {"second_chance": True, "motion_confirm": True},
    "qw_second": {"learn_gate": True, "fall_size": True, "lost_focus": True, "second_chance": True, "motion_confirm": True},
    # deep dive: identity (lite re-ID on plain crops needs its own calibration: 0.47/0.78)
    "negtxn": {"negatives": True, "transact": True},
    "plaincal": {"plain_reid": True, "reid_lo": 0.47, "reid_hi": 0.78},
    "plaincal_neg": {"plain_reid": True, "reid_lo": 0.47, "reid_hi": 0.78, "negatives": True},
    "plaincal_negtxn": {"plain_reid": True, "reid_lo": 0.47, "reid_hi": 0.78, "negatives": True, "transact": True},
    # deep dive: tie-in and ground
    "ground": {"ground_latch": True, "ground_handheld": True},
    "tiein": {"ground_latch": True, "ground_handheld": True, "tiein_s": 60.0},
    # everything that looked good alone, together
    "combo": {"learn_gate": True, "fall_size": True, "lost_focus": True,
              "second_chance": True, "motion_confirm": True,
              "negatives": True, "transact": True,
              "ground_latch": True, "ground_handheld": True, "tiein_s": 60.0},
    "combo_plain": {"learn_gate": True, "fall_size": True, "lost_focus": True,
                    "second_chance": True, "motion_confirm": True,
                    "plain_reid": True, "reid_lo": 0.47, "reid_hi": 0.78, "negatives": True, "transact": True,
                    "ground_latch": True, "ground_handheld": True, "tiein_s": 60.0},
}
RUNNER_KEYS = {"lost_focus", "second_chance", "motion_confirm", "plain_reid", "ground_handheld"}


def _stamp(cfg: Config, run_opts: dict, referee) -> dict:
    """What produced a run: code version and every setting, so results stay traceable."""
    import subprocess
    from dataclasses import asdict
    git = subprocess.run(["git", "-C", os.getcwd(), "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
    return {"git": git, "pipeline": pipeline.current().name, "variant": os.environ.get("DIRECTE_VARIANT", ""),
            "config": asdict(cfg), "runner": run_opts, "focus": FOCUS, "focus_k": FOCUS_K,
            "referee": None if referee is None else {"model": referee.model, "policy": asdict(referee.policy)}}


def suffix_for(model: str | None) -> str:
    v = os.environ.get("DIRECTE_VARIANT", "")
    return ("" if not model else "_ref_" + model.split("/")[-1]) + (f"_{v}" if v else "")


def _suffix(referee) -> str:
    return suffix_for(None if referee is None else referee.model)


def load(clip: Clip, suffix: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(clip.run_dir / f"online_{pipeline.current().name}{suffix}.npz")
    return z["t"], z["state"], z["box"]


def score_samples(clip: Clip, samples: list[dict], suffix: str = "") -> dict:
    """Score an online run against checked samples ({t, box}) by nearest tick in time."""
    t, state, box = load(clip, suffix)
    c = {"right": 0, "wrong": 0, "missed": 0, "false_lock": 0, "absent_ok": 0}
    for s in samples:
        if s.get("skip") or not len(t):
            continue
        k = int(np.argmin(np.abs(t - s["t"])))
        if abs(t[k] - s["t"]) > 0.08:
            continue
        locked = state[k] == LOCKED
        if s["box"] is None:
            c["false_lock" if locked else "absent_ok"] += 1
        elif not locked:
            c["missed"] += 1
        elif iou(box[k], np.array([s["box"]]))[0] >= 0.3:
            c["right"] += 1
        else:
            c["wrong"] += 1
    return c
