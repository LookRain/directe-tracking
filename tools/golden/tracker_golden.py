"""Golden fixtures for the Swift TrackerCore module (Sources/TrackerCore): the Python tracker
(python/directe_tracking/tracker.py tag + TargetTracker) replayed online-style on small synthetic scenes, for the
Swift tests to replay bit-exactly.

  uv run --project python python tools/golden/tracker_golden.py

Every case runs:
  1. tag window = the first tick, tag() on it around the tap;
  2. TargetTracker(profile, box, t_k, cfg, frame_h=H); step every window tick (no camera call), then
     per tick: camera(shift, conf) -> predicted_box(t) -> [actions] -> step(t, cands[, spotter]).
Actions stand in for a referee, recorded per tick as (veto, force, flags): veto = index of a view
candidate to exclude for this step, force = index of the candidate to force() onto (its box as
float64, as an app passes boxes), flags bit 0 = release(t), bit 1 = allow_reacquire off.

Synthetic scenes (float32 boxes, 8-d embeddings): the selftest's scenes (fall at 10/5 fps, lookalike, camera bumps,
tie-in slid/queue) plus a tiny climber (height < 30 px), a climber tagged with the feet at the frame bottom who
climbs up and back down past a belayer (ground latch on and off, empty frames), an embedding that drifts toward a
banked negative, a spotter, and a look-alike stacked below the climber (Config.switch_margin). Configs: default,
combo, combo_alt (combo + reid_mode fallback, others_always, no transact) and colour_tiein (colour only, tiein_s=60,
as the selftest's tie-in); the stacked scene adds unguarded (switch_margin off) and no_keep (track_keep_s 0).

The same machinery records cases on real footage (`write(case, out)`, cases built from cached detections); those
stay with whoever filmed them, and the Swift tests pick them up from $DIRECTE_GOLDEN.

Writes Tests/TrackerCoreTests/Fixtures/tracker_<case>.{json,bin}: the .bin is raw DEFLATE (zlib wbits=-15) of the
named arrays the .json indexes (dtype, shape, byte offset); every number the Swift test compares lives in the .bin
(JSON parsers needn't round-trip doubles exactly).
"""

from __future__ import annotations

import json
import sys
import zlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from directe_tracking import appearance  # noqa: E402
from directe_tracking import tracker as T  # noqa: E402

OUT = REPO / "Tests/TrackerCoreTests/Fixtures"
COMBO = dict(learn_gate=True, fall_size=True, negatives=True, transact=True, ground_latch=True, tiein_s=60.0)
STATE_CODE = {T.LOCKED: 0, T.UNCERTAIN: 1, T.LOST: 2}
NO_ACTION = (-1, -1, 0)

Actions = Callable[[int, float, T.Cands, int], tuple[int, int, int]]


def referee(n: int, t: float, c: T.Cands, last: int) -> tuple[int, int, int]:
    """Segments' referee stand-in, by tick number: veto the current lock for a tick, drop the lock,
    force a lock onto the last candidate, keep re-locks off for a while."""
    veto = last if n % 37 == 5 and last >= 0 else -1
    force = len(c) - 1 if n % 71 == 40 and len(c) else -1
    flags = (1 if n % 53 == 20 else 0) | (2 if 60 <= n % 97 < 70 else 0)
    return veto, force, flags


def apply(trk: T.TargetTracker, act, t: float, c: T.Cands, trust_s: float) -> None:
    veto, force, flags = act
    trk.exclude = None if veto < 0 else np.arange(len(c)) == veto
    trk.allow_reacquire = not flags & 2
    if flags & 1:
        trk.release(t)
    if force >= 0:
        trk.force(c.boxes[force].astype(np.float64), t, trust_s=trust_s)


@dataclass
class Case:
    name: str
    kind: str
    ts: list[float]
    boxes: np.ndarray      # float32 [N, 4]
    feats: np.ndarray      # float16 [N, 4, 17]
    valids: np.ndarray     # bool [N, 4]
    embs: np.ndarray       # float16 [N, K]
    offsets: np.ndarray    # int32 [T + 1]
    shift: np.ndarray      # float64 [T, 2]
    conf: np.ndarray       # float64 [T]
    W: int
    H: int
    seed: tuple[float, float]  # normalised tap
    n_win: int
    configs: dict[str, tuple[T.Config, Actions | None]]
    spotter: bool = False
    trust_s: float = 0.0
    meta: dict = field(default_factory=dict)

    def cands(self, n: int) -> T.Cands:
        a, b = self.offsets[n], self.offsets[n + 1]
        return T.Cands(self.boxes[a:b], self.feats[a:b].astype(np.float32), self.valids[a:b],
                       self.embs[a:b].astype(np.float32))


def spotter_of(c: T.Cands) -> T.Cands:
    """A second camera's view of the same people (synthetic: 3 px right and down)."""
    return T.Cands(c.boxes + np.float32(3.0), c.feats, c.valids, c.embs)


class Window:
    """Just enough of Detections for tracker.tag() (as lab/online.py _Fake)."""

    def __init__(self, ts, cands, W, H):
        self.t = np.array(ts)
        self.size = (W, H)
        self.offsets = np.cumsum([0] + [len(c) for c in cands])
        self.boxes = np.concatenate([c.boxes for c in cands])
        self.feat = np.concatenate([c.feats for c in cands])
        self.fvalid = np.concatenate([c.valids for c in cands])
        setattr(self, T.EMB_KEY, np.concatenate([c.embs for c in cands]))

    def __len__(self):
        return len(self.t)

    def frame(self, i):
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))


def replay(case: Case, cfg: T.Config, actions: Actions | None):
    win = [case.cands(n) for n in range(case.n_win)]
    profile, box, k = T.tag(Window(case.ts[:case.n_win], win, case.W, case.H),
                            (case.ts[0], float(case.seed[0]), float(case.seed[1])))
    trk = T.TargetTracker(profile, box, float(case.ts[k]), cfg, frame_h=case.H)
    rows, last = [], -1
    for n, t in enumerate(case.ts):
        c = case.cands(n)
        pred, act = np.full(4, np.nan), NO_ACTION
        if n >= case.n_win:
            trk.camera(case.shift[n], float(case.conf[n]))
            pred = trk.predicted_box(t)
            if actions is not None:
                act = actions(n, t, c, last)
                apply(trk, act, t, c, case.trust_s)
        s = trk.step(t, c, spotter_of(c) if case.spotter else None)
        last = s.chosen
        flags = (int(trk.jump_t == t) | int(trk.unstable) << 1 | int(trk.latched) << 2
                 | int(trk.tiein) << 3 | int(trk.shift_unknown) << 4)
        p = trk.p
        rows.append(dict(state=STATE_CODE[s.state], chosen=s.chosen, box=np.asarray(s.box, np.float64),
                         app=s.app, score=s.score, vel=np.asarray(s.velocity, np.float64), pred=pred,
                         flags=flags, ground=float(trk.ground), app_level=float(trk.app_level),
                         counts=(len(p.gallery), len(p.negs), len(p.pending), len(trk.people)), act=act,
                         source=int(s.source == "spotter")))
    return (profile, box, k), rows, trk


def write(case: Case, out: Path = OUT) -> int:
    arrays: dict[str, np.ndarray] = {
        "boxes": case.boxes, "feats": case.feats, "valids": case.valids.astype(np.uint8), "embs": case.embs,
        "offsets": case.offsets, "t": np.array(case.ts, np.float64), "shift": case.shift.astype(np.float64),
        "conf": case.conf.astype(np.float64), "tap": np.array([case.seed[0] * case.W, case.seed[1] * case.H]),
    }
    meta = {"name": case.name, "kind": case.kind, "width": case.W, "height": case.H,
            "window_ticks": case.n_win, "spotter": case.spotter, "trust_s": case.trust_s, **case.meta,
            "configs": [], "results": {}}
    for name, (cfg, actions) in case.configs.items():
        (profile, box, k), rows, trk = replay(case, cfg, actions)
        if "tag_index" not in meta:
            meta["tag_index"] = k
            arrays["tag/box"] = np.asarray(box, np.float64)
            arrays["tag/feat"] = profile.feat.astype(np.float32).ravel()
            arrays["tag/valid"] = profile.valid.astype(np.uint8)
            arrays["tag/emb"] = profile.emb.astype(np.float32)
        meta["configs"].append({"name": name, "config": asdict(cfg)})
        for f in ("state", "chosen", "app", "score", "flags", "ground", "app_level", "source"):
            dt = {"state": np.uint8, "chosen": np.int16, "flags": np.uint8, "source": np.uint8}.get(f, np.float64)
            arrays[f"{name}/{f}"] = np.array([r[f] for r in rows], dt)
        for f in ("box", "vel", "pred"):
            arrays[f"{name}/{f}"] = np.stack([r[f] for r in rows]).astype(np.float64)
        arrays[f"{name}/counts"] = np.array([r["counts"] for r in rows], np.int16)
        arrays[f"{name}/act"] = np.array([r["act"] for r in rows], np.int16)
        st, fl, cnt = arrays[f"{name}/state"], arrays[f"{name}/flags"], arrays[f"{name}/counts"]
        res = {
            "locked": int((st == 0).sum()), "uncertain": int((st == 1).sum()), "lost": int((st == 2).sum()),
            "relocks": int(((st[:-1] == 2) & (st[1:] == 0)).sum()), "jump_refusals": int((fl & 1).sum()),
            "unstable": int(((fl >> 1) & 1).sum()), "latched": int(((fl >> 2) & 1).sum()),
            "max_gallery": int(cnt[:, 0].max()), "max_negs": int(cnt[:, 1].max()), "max_pending": int(cnt[:, 2].max()),
            "events": [[float(e[0]), e[1]] for e in trk.events],
        }
        meta["results"][name] = res
        print(f"[golden] {case.name} {name}: {res}", flush=True)
    blob, index = bytearray(), {}
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        index[name] = {"dtype": a.dtype.str.lstrip("<|"), "shape": list(a.shape), "offset": len(blob)}
        blob += a.tobytes()
    meta["arrays"] = index
    meta["raw_bytes"] = len(blob)
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    packed = co.compress(bytes(blob)) + co.flush()
    (out / f"tracker_{case.name}.bin").write_bytes(packed)
    (out / f"tracker_{case.name}.json").write_text(json.dumps(meta, indent=1))
    print(f"[golden] {case.name}: {len(case.ts)} ticks, {len(case.boxes)} candidates, "
          f"{len(packed) / 1e6:.3f} MB ({len(blob) / 1e6:.2f} raw)", flush=True)
    return len(packed)


# ---- synthetic scenes

SB = 200.0  # body height, px
K_SYN = 8
UNIFORM = np.full((len(appearance.REGIONS), appearance.DIM), 1.0 / appearance.DIM, np.float32)


def unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def axis(*xs) -> np.ndarray:
    return unit(list(xs) + [0.0] * (K_SYN - len(xs)))


EA, EB, ED = axis(1.0), axis(0.5, 0.866), axis(0.8, 0.0, 0.6)  # climber, someone else, a re-ID lookalike


def colour(*bins) -> np.ndarray:
    f = np.zeros((len(appearance.REGIONS), appearance.DIM), np.float32)
    for k, w in bins:
        f[:, k] = w
    return f


def person(x, y, emb, h=SB, feat=UNIFORM, valid=None) -> dict:
    return {"box": (x - 0.3 * h, y - h / 2, x + 0.3 * h, y + h / 2), "emb": emb, "feat": feat,
            "valid": np.ones(len(appearance.REGIONS), bool) if valid is None else valid}


SYNTH_CONFIGS = {
    "default": {},
    "combo": COMBO,
    "combo_alt": {**COMBO, "reid_mode": "fallback", "others_always": True, "transact": False},
    "colour_tiein": {"use_reid": False, "tiein_s": 60.0},
}


def synthetic(name, frames, W=1920, H=2160, shift=None, conf=None, actions=None, spotter=False,
              trust_s=0.0, configs=None, desc="") -> Case:
    rng = np.random.default_rng(sum(map(ord, name)))  # deterministic per scene
    ts = [t for t, _ in frames]
    people = [p for _, ps in frames for p in ps]
    embs = [unit(p["emb"] + rng.normal(0, 0.02, K_SYN)) for p in people]
    first = frames[0][1][0]["box"]
    seed = ((first[0] + first[2]) / 2 / W, (first[1] + first[3]) / 2 / H)
    cfgs = {k: (replace(T.Config(), **v), actions) for k, v in (configs or SYNTH_CONFIGS).items()}
    n = len(frames)
    return Case(
        name=f"synth_{name}", kind="synthetic", ts=ts,
        boxes=np.array([p["box"] for p in people], np.float32).reshape(-1, 4),
        feats=np.array([p["feat"] for p in people], np.float16).reshape(-1, len(appearance.REGIONS), appearance.DIM),
        valids=np.array([p["valid"] for p in people], bool).reshape(-1, len(appearance.REGIONS)),
        embs=np.array(embs, np.float16).reshape(-1, K_SYN),
        offsets=np.cumsum([0] + [len(ps) for _, ps in frames]).astype(np.int32),
        shift=np.zeros((n, 2)) if shift is None else np.asarray(shift, np.float64),
        conf=np.ones(n) if conf is None else np.asarray(conf, np.float64),
        W=W, H=H, seed=seed, n_win=1, configs=cfgs, spotter=spotter, trust_s=trust_s, meta={"desc": desc})


def scenes() -> list[Case]:
    out = []
    g = 9.8 / 1.7 * SB  # px/s^2 for a 1.7 m climber
    for fps in (10, 5):
        frames = []
        for k in range(int(3 * fps)):
            t = k / fps
            tf = min(max(t - 1.0, 0.0), 1.2)
            frames.append((t, [person(500, 600 + 0.5 * g * tf ** 2, EA), person(500 - 3 * SB, 600 + 3 * SB, EB)]))
        out.append(synthetic(f"fall{fps}", frames, desc="selftest: 1.2 s free fall, a bystander below"))

    def lookalike(t):
        hidden = 1.0 <= t < 2.5
        return ([] if hidden else [person(500, 600, EA)]) + [person(500 - 1.3 * SB, 600, ED)]
    out.append(synthetic("lookalike", [(k * 0.1, lookalike(k * 0.1)) for k in range(40)],
                         desc="selftest: climber hidden 1.5 s, an identical person 1.3 heights to the side"))
    out.append(synthetic("spotter", [(k * 0.1, lookalike(k * 0.1)) for k in range(40)], spotter=True,
                         actions=lambda n, t, c, last: (-1, -1, 1 if n in (20, 33) else 0),
                         desc="lookalike, with a second camera (spotter) and two releases"))

    for known in (True, False):
        frames, shift, conf, off = [], [], [], np.zeros(2)
        for k in range(40):
            s = np.array([300.0, -250.0]) if k == 15 else np.zeros(2)  # someone knocks the phone
            off = off + s
            shift.append(s if known else np.zeros(2))
            conf.append(0.9 if known or k != 15 else 0.0)
            frames.append((k * 0.1, [person(500 + off[0], 600 + off[1], EA),
                                     person(500 - 1.3 * SB + off[0], 600 + off[1], EB)]))
        out.append(synthetic(f"bump_{'known' if known else 'unknown'}", frames, shift=shift, conf=conf,
                             desc="selftest: camera bump, shift measured or unknown"))

    A, B = colour((0, 1.0)), colour((0, 0.6), (5, 0.4))
    for slid in (True, False):
        frames, a, b = [], (500.0, 600.0), (660.0, 600.0)
        for k in range(80):
            t = k * 0.1
            if t >= 2.0:
                if slid:
                    a = (560.0, 600.0 - SB * (t - 2.0))
                else:
                    b = (660.0, 600.0 - SB * (t - 2.0))
            hidden = slid and 1.0 <= t < 2.0
            frames.append((t, ([] if hidden else [person(*a, EA, feat=A)]) + [person(*b, EB, feat=B)]))
        acts = (lambda n, t, c, last: (-1, 0 if n == 10 else -1, 0)) if slid else None  # t = 1.0: onto the belayer
        out.append(synthetic(f"tiein_{'slid' if slid else 'queue'}", frames, actions=acts,
                             desc="selftest: tie-in, the lock slid onto the belayer / a queue"))

    frames = []
    for k in range(40):
        x = 500 + 5 * k
        ps = [] if 15 <= k < 20 else [person(x, 600, EA, h=20)]
        frames.append((k * 0.1, ps + [person(x + 40, 600, EB, h=20), person(x - 60, 610, ED, h=24)]))
    out.append(synthetic("tiny", frames, desc="a climber under 30 px tall (h_ref is the Python float 30.0)"))

    frames, H = [], 1080
    for k in range(180):
        t = k * 0.1
        up = 100 * min(max(t - 2, 0), 7) - 100 * min(max(t - 9, 0), 7)  # up 700 px over 7 s, then back down
        ps = [] if 50 <= k < 53 else [person(900, H - 1 - 150 - up, EA, h=300), person(1150, H - 1 - 150, EB, h=300)]
        if 60 <= k < 120:
            ps.append(person(300 + 5 * (k - 60), H - 1 - 140, ED, h=280))  # someone walks past at the base
        frames.append((t, ps))
    out.append(synthetic("bottom", frames, H=H, desc="tagged with the feet at the frame bottom; climbs 2.3 heights "
                         "and back down beside the belayer; empty frames; a walker at the base"))

    frames = []
    for k in range(100):
        t = k * 0.1
        a = min(max((t - 3.0) / 4.0, 0.0), 0.9)
        frames.append((t, [person(500 + 10 * np.sin(t), 600, unit((1 - a) * EA + a * EB)),
                           person(500 + 2.5 * SB, 600, EB), person(500 - 4 * SB, 600, ED)]))
    out.append(synthetic("drift", frames, desc="the climber's embedding drifts toward a banked negative"))

    frames = []
    off = colour((3, 1.0))
    for k in range(75):
        t = k * 0.1
        y = 1200 - 0.25 * SB * t
        h, emb, feat = SB, EA, UNIFORM
        if 3.0 <= t < 3.2:
            feat = off                    # the climber's colour reads wrong
        if 4.5 <= t < 4.8:
            emb = unit(0.1 * EA + 0.9 * EB)  # ... their re-ID view too
        if 6.6 <= t < 6.9:
            h, emb = 0.6 * SB, unit(0.25 * EA + 0.75 * EB)  # folded up and turned: the look-alike now wins clearly
        ps = [] if 2.0 <= t < 2.4 or 5.5 <= t < 5.7 else [person(500, y, emb, h=h, feat=feat)]
        if abs(t - 2.6) < 1e-6 or abs(t - 5.0) < 1e-6:
            ps.append(person(500, y - 0.2 * SB, EA, h=0.6 * SB))  # a second box: the climber's upper body
        if not abs(t - 1.9) < 1e-6:  # the look-alike, missed once; it closes in below, then beside
            ps.append(person(540, y + 0.8 * SB, ED) if t < 4.0 else
                      person(540, y + 0.35 * SB, ED) if t < 6.0 else person(560, y + 0.1 * SB, ED))
        frames.append((t, ps))
    out.append(synthetic("stacked", frames, configs={**SYNTH_CONFIGS, "unguarded": {"switch_margin": -1.0, "track_keep_s": 0.0},
                                                     "no_keep": {"track_keep_s": 0.0}},
                         desc="a look-alike a body length below on the next route (steep side view, boxes overlap): "
                         "missed detections, colour and re-ID wobbles, a second box of the climber"))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for case in scenes():
        total += write(case)
    print(f"[golden] total {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
