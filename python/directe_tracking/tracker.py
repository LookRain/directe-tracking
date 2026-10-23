"""Target tracker: keeps one tagged climber across detections, with Locked/Uncertain/Lost states.

Works in "world" pixel coordinates: the wide source frame. Offline that's just the frame; in the
simulator, detections from the virtual camera are mapped back into it using the camera pose, the
way the app will use the phone's motion sensors.

Identity uses two cues when available:
  - colour signatures of body regions (lab/appearance.py), sampled inside the person mask;
  - a learned re-identification embedding (OSNet-AIN, lab/identity.py): an anchor from the tag
    moment plus a small gallery of other viewpoints collected while confidently locked.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from . import appearance

LOCKED, UNCERTAIN, LOST = "locked", "uncertain", "lost"
EMB_KEY = "emb_masked"  # which embedding to use: "emb" (plain crop) or "emb_masked" (background greyed)


@dataclass
class Config:
    app_min: float = 0.50          # absolute appearance floor to accept while locked/uncertain
    app_rel: float = 0.75          # ...and relative to how well the target has been matching
    accept: float = 0.55           # combined score to accept
    reacquire_app: float = 0.75    # appearance needed to start a re-lock while lost
    reacquire_frames: int = 3      # consecutive consistent sightings to re-lock
    uncertain_s: float = 2.0       # predict-only time before declaring lost
    adapt_rate: float = 0.03       # colour profile adaptation per confident frame
    adapt_app: float = 0.70
    use_colour: bool = True
    use_reid: bool = True
    colour_weight: float = 0.2     # share of colour in the combined appearance (rest: re-ID)
    reid_mode: str = "blend"       # "blend": mix with colour; "fallback": only when colour can't judge
    reid_lo: float = 0.35          # cosine mapped to 0 ...
    reid_hi: float = 0.75          # ... and to 1
    gallery_size: int = 30
    gallery_every_s: float = 1.0
    max_sigma: float = 3.0         # widest position window (body heights) while uncertain
    ground_rule: bool = True       # once the climber is up the wall, nobody at ground level is them
    max_size_ratio: float = 2.5    # a candidate this much bigger/smaller than the climber is someone else
    # Jump guard: a lock may move to a box that doesn't touch the climber's predicted box only as a
    # plausible fall, and never onto someone who was already standing/climbing elsewhere.
    # Measured on the 8 clips: real motion never left a gap (fastest 0.76 body heights/frame,
    # box-shape noise); every switch to another person did (1.1-4 body heights in one frame).
    jump_guard: bool = True
    move_rate: float = 1.5         # body heights/s any direction (climbing, swinging) beyond touching
    fall_accel: float = 11.5       # body heights/s^2, ~2 g at 1.7 m per body height
    jump_margin: float = 0.1       # body heights of slack
    settle_s: float = 0.5          # seen this long as a separate person = someone else
    others_always: bool = False    # ...and someone else can't take the lock even when their box touches
    unstable_shift: float = 1.0    # camera moved this many body heights in a frame: freeze identity
    # Ground rule made independent of the lock being right (docs/review/deep-dives/ascent-and-ground.md):
    # "the climber is up" holds through a loss and is released only by following them down; the
    # ground line moves with the camera; tagged mid-wall, someone standing beside the tag gives the ground.
    ground_latch: bool = False
    # Tie-in phase: the climber is whoever leaves the ground. If the lock still stands at the tag's
    # ground level while someone who appeared beside it climbs (feet and head well above the lock's)
    # and matches the *tag-time* profile clearly better than the lock does, re-lock onto them.
    tiein_s: float = 0.0           # watch this long after the tag (0: off; 60 in the deep dive)
    riser_feet_h: float = 1.0      # riser's feet this many own heights above the lock's feet...
    riser_head_h: float = 0.5      # ...head this many above the lock's head (someone standing farther
                                   # back has higher feet but a lower head)
    riser_size: float = 0.25       # ...at least this fraction of the lock's height
    riser_column: float = 1.5      # ...centre within this many lock widths
    riser_hold_s: float = 1.5      # ...all of it for this long
    riser_margin: float = 0.15     # appearance vs the tag-time profile: riser >= lock + margin...
    riser_floor: float = 0.5       # ...and >= floor
    # Learn the profile only from steady views: not during falls, fast moves or camera motion, and
    # not right after the guard refused a jump (a blurred mid-fall view once became a profile image).
    learn_gate: bool = False
    learn_max_speed: float = 1.0   # body heights/s
    # After a fall toward the camera the climber's box grows 4-6x; judge size against that.
    fall_size: bool = False
    fall_size_ratio: float = 6.0   # max size ratio within fall_size_s of fast downward motion
    fall_size_s: float = 4.0
    fall_speed: float = 2.0        # body heights/s downward = falling
    # Learn who it isn't, and make learning undoable (docs/review/deep-dives/identity-and-product.md A2).
    negatives: bool = False        # bank other settled people seen next to a confident lock
    neg_lam: float = 1.0           # re-ID cos -= lam * max(0, cos(nearest negative) - cos(profile) + margin)
    neg_margin: float = 0.05
    neg_colour: bool = True        # body regions a negative shares with the profile count less
    neg_every_s: float = 0.5
    neg_size: int = 60
    transact: bool = False         # profile updates wait commit_s with no doubt; any doubt discards them
    commit_s: float = 5.0
    doubt_rival: float = 0.05      # a rival this close is a doubt


@dataclass
class Cands:
    """Candidate people for one frame (world boxes)."""

    boxes: np.ndarray            # [N,4]
    feats: np.ndarray            # [N,4,DIM]
    valids: np.ndarray           # [N,4]
    embs: np.ndarray | None = None  # [N,E] L2-normalised

    def __len__(self) -> int:
        return len(self.boxes)

    @classmethod
    def of(cls, dets, sl: slice) -> "Cands":
        embs = getattr(dets, EMB_KEY, None)
        return cls(dets.boxes[sl], dets.feat[sl], dets.fvalid[sl], None if embs is None else embs[sl])

    def subset(self, keep: np.ndarray) -> "Cands":
        return Cands(self.boxes[keep], self.feats[keep], self.valids[keep],
                     None if self.embs is None else self.embs[keep])

    def concat(self, other: "Cands") -> "Cands":
        embs = None
        if self.embs is not None and other.embs is not None:
            embs = np.concatenate([self.embs.reshape(-1, other.embs.shape[1]), other.embs])
        return Cands(np.concatenate([self.boxes.reshape(-1, 4), other.boxes]),
                     np.concatenate([self.feats.reshape(-1, *other.feats.shape[1:]), other.feats]),
                     np.concatenate([self.valids.reshape(-1, other.valids.shape[1]), other.valids]),
                     embs)


@dataclass
class Profile:
    feat: np.ndarray    # [4, DIM] colour, from the tag moment; never changes
    valid: np.ndarray   # [4]
    emb: np.ndarray | None = None   # [E] re-ID anchor from the tag moment
    adapt: np.ndarray = field(default=None)  # colour copy that slowly follows lighting/viewpoint
    avalid: np.ndarray = field(default=None)
    gallery: list = field(default_factory=list)  # re-ID embeddings of other viewpoints
    last_gallery_t: float = -1e9
    negs: list = field(default_factory=list)     # re-ID embeddings of people who aren't the climber
    neg_f: list = field(default_factory=list)    # ...and their colour
    neg_v: list = field(default_factory=list)
    pending: list = field(default_factory=list)  # (t, kind, payload) updates waiting to be committed
    last_neg_t: float = -1e9

    def __post_init__(self):
        if self.adapt is None:
            self.adapt = self.feat.copy()
            self.avalid = self.valid.copy()

    def colour(self, c: Cands, cfg: Config | None = None) -> np.ndarray:
        if cfg is not None and cfg.negatives and cfg.neg_colour and self.neg_f:
            # A region a known non-climber shares with the profile (the belayer's blue shorts too)
            # stops counting much.
            F, V = np.array(self.neg_f), np.array(self.neg_v)
            wm = np.ones(len(appearance.REGIONS))
            for prof, pv in ((self.feat, self.valid), (self.adapt, self.avalid)):
                bc = np.where(V & pv[None], np.sqrt(F * prof[None]).sum(axis=2), 0.0).max(axis=0)
                wm = np.minimum(wm, np.clip((0.95 - bc) / 0.35, 0.15, 1.0))
            a = _sim_weighted(self.feat, self.valid, c.feats, c.valids, wm)
            b = _sim_weighted(self.adapt, self.avalid, c.feats, c.valids, wm)
            return np.maximum(a, b)
        a = appearance.similarity_many(self.feat, self.valid, c.feats, c.valids)
        b = appearance.similarity_many(self.adapt, self.avalid, c.feats, c.valids)
        return np.maximum(a, b)

    def pos_cos(self, c: Cands) -> np.ndarray | None:
        if self.emb is None or c.embs is None or len(c) == 0:
            return None
        cos = c.embs @ self.emb
        if self.gallery:
            g = (c.embs @ np.array(self.gallery).T).max(axis=1)
            cos = np.maximum(cos, g - 0.03)  # gallery views count, slightly less than the anchor
        return cos

    def neg_cos(self, c: Cands) -> np.ndarray | None:
        if not self.negs or c.embs is None or len(c) == 0:
            return None
        return (c.embs @ np.array(self.negs).T).max(axis=1)

    def reid_cos(self, c: Cands, cfg: Config | None = None) -> np.ndarray | None:
        pos = self.pos_cos(c)
        if pos is None or cfg is None or not cfg.negatives:
            return pos
        neg = self.neg_cos(c)
        if neg is None:
            return pos
        return pos - cfg.neg_lam * np.maximum(0.0, neg - pos + cfg.neg_margin)  # nearer a non-climber: penalise

    def sim(self, c: Cands, cfg: Config) -> np.ndarray:
        """Appearance per candidate in [0, 1], or UNKNOWN."""
        colour = self.colour(c, cfg) if cfg.use_colour else np.full(len(c), appearance.UNKNOWN)
        cos = self.reid_cos(c, cfg) if cfg.use_reid else None
        if cos is None:
            return colour
        r = np.clip((cos - cfg.reid_lo) / (cfg.reid_hi - cfg.reid_lo), 0.0, 1.0)
        if cfg.reid_mode == "fallback":
            return np.where(colour >= 0, colour, r)
        w = cfg.colour_weight
        return np.where(colour >= 0, w * colour + (1 - w) * r, r)

    def learn(self, c: Cands, k: int, t: float, cfg: Config) -> None:
        f, v = c.feats[k].copy(), c.valids[k].copy()
        if cfg.transact:
            self.pending.append((t, "colour", (f, v)))
        else:
            self._adapt_colour(f, v, cfg)
        if c.embs is None or self.emb is None or t - self.last_gallery_t < cfg.gallery_every_s:
            return
        e = c.embs[k]
        if float(e @ self.emb) < cfg.reid_lo + 0.1:
            return  # too far from the anchor to trust as the same person
        if self.gallery and float(np.max(np.array(self.gallery) @ e)) > 0.9:
            return  # nothing new
        self.last_gallery_t = t
        if cfg.transact:
            self.pending.append((t, "gallery", e.copy()))
        else:
            self._add_gallery(e.copy(), cfg)

    def _adapt_colour(self, f: np.ndarray, v: np.ndarray, cfg: Config) -> None:
        for r in range(len(v)):
            if v[r]:
                if self.avalid[r]:
                    self.adapt[r] = (1 - cfg.adapt_rate) * self.adapt[r] + cfg.adapt_rate * f[r]
                else:
                    self.adapt[r], self.avalid[r] = f[r], True

    def _add_gallery(self, e: np.ndarray, cfg: Config) -> None:
        self.gallery.append(e)
        if len(self.gallery) > cfg.gallery_size:
            self.gallery.pop(0)

    def add_negative(self, e: np.ndarray, f: np.ndarray, v: np.ndarray, t: float, cfg: Config) -> None:
        if self.negs and float(np.max(np.array(self.negs) @ e)) > 0.92:
            return  # nothing new
        item = (e.copy(), f.copy(), v.copy())
        if cfg.transact:
            self.pending.append((t, "neg", item))
        else:
            self._commit_neg(item, cfg)

    def _commit_neg(self, item, cfg: Config) -> None:
        e, f, v = item
        self.negs.append(e); self.neg_f.append(f); self.neg_v.append(v)
        if len(self.negs) > cfg.neg_size:
            for lst in (self.negs, self.neg_f, self.neg_v):
                lst.pop(0)

    def commit_due(self, t: float, cfg: Config) -> None:
        """Updates that survived commit_s without a doubt become part of the profile."""
        keep = []
        for item in self.pending:
            t0, kind, payload = item
            if t - t0 < cfg.commit_s:
                keep.append(item)
            elif kind == "colour":
                self._adapt_colour(*payload, cfg)
            elif kind == "gallery":
                self._add_gallery(payload, cfg)
            else:
                self._commit_neg(payload, cfg)
        self.pending = keep

    def doubt(self) -> None:
        """Something suggests the lock may have been wrong lately: forget what it taught us."""
        self.pending = []


def _sim_weighted(profile, pvalid, feats, valids, wmul) -> np.ndarray:
    """appearance.similarity_many with per-region weight multipliers."""
    both = valids & pvalid[None]
    bc = np.sqrt(feats * profile[None]).sum(axis=2)
    w = (appearance.WEIGHTS * wmul)[None] * both
    evidence = (appearance.WEIGHTS[None] * both).sum(axis=1)  # judged on the base weights
    mean = (bc * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-6)
    worst = np.where(both & (appearance.WEIGHTS[None] >= 0.2) & (wmul[None] >= 0.5), bc, 1.0).min(axis=1)
    s = np.clip(mean - appearance.VETO_GAIN * np.maximum(0.0, appearance.VETO_BELOW - worst), 0.0, 1.0)
    return np.where(evidence >= appearance.MIN_EVIDENCE, s, appearance.UNKNOWN)


@dataclass
class Step:
    state: str
    box: np.ndarray        # [4] world xyxy (predicted when not locked)
    velocity: np.ndarray   # [2] world px/s
    chosen: int            # index into this frame's candidates, -1 if none
    app: float
    score: float
    source: str = "view"   # "view" (the recording camera) or "spotter" (the wide camera)


@dataclass(eq=False)
class _Person:
    """Short-lived IoU track of anyone (tie-in phase and the tagged-on-the-wall check)."""
    t0: float
    t: float
    box: np.ndarray
    box0: np.ndarray                  # first box
    lock0: np.ndarray                 # the lock's box when this person first appeared
    up_since: float | None = None
    apps: list = field(default_factory=list)  # appearance vs the tag-time profile


class TargetTracker:
    def __init__(self, profile: Profile, box: np.ndarray, t: float, cfg: Config | None = None,
                 frame_h: float | None = None):
        self.p = profile
        self.cfg = cfg or Config()
        self.c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float64)
        self.wh = np.array([box[2] - box[0], box[3] - box[1]], dtype=np.float64)
        self.v = np.zeros(2)
        self.t = t
        self.t_seen = t
        self.state = LOCKED
        self.app_level = 0.85  # running average of the target's appearance match
        self.ground = float(box[3])  # feet at the tag moment: ground level (fixed camera / world frame)
        self.h0 = float(box[3] - box[1])
        self._pending: tuple[np.ndarray, int] | None = None
        self.exclude: np.ndarray | None = None  # per-candidate veto for the next step (referee)
        self.rival_gap: float | None = None     # score margin over the next-best person, last match
        self.allow_reacquire = True             # the referee turns this off: it decides re-locks
        self.trust_until = -1e9                 # after a referee confirmation, follow by position
        self.unstable = False                   # camera being bumped/handled: no identity changes
        self.shift_unknown = False
        self.jump_t = -1e9                      # last time the jump guard refused a move
        self.fall_t = -1e9                      # last time the climber moved down fast
        self._tracks = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, bool))  # last frame: boxes, age (s), is the target
        if self.cfg.ground_latch and frame_h is not None and box[3] >= frame_h - 3:
            self.ground = float(frame_h)        # feet below the frame: the ground is at least the bottom edge
        self.frame_h = frame_h
        self.t_tag, self.ground_tag = t, self.ground
        self.tag_profile = copy.deepcopy(profile)  # the tie-in compares against this, never the live profile
        self.tiein = self.cfg.tiein_s > 0
        self.people: list[_Person] = []
        self._pid = np.zeros(0, int)
        self._lock_apps: list = []
        self.latched = False                    # ground rule: "the climber is up", see _latch
        self.cont = True                        # lock continuous since the tag / last force()
        self._up_t, self._jumped_since_up, self._prev = -1e9, False, None
        self.events: list[tuple] = []

    def camera(self, shift: np.ndarray, conf: float = 1.0) -> None:
        """The camera itself moved the image by `shift` px since the last frame (on the phone: gyro
        and gimbal angles; in the lab: image registration). Move what we know with it, so people's
        motion is judged against the wall, not the screen. A violent or unreadable move (bump,
        someone picking the phone up) freezes identity changes until it settles."""
        shift = np.asarray(shift, np.float64)
        self.c = self.c + shift
        b, a, tg = self._tracks
        if len(b):
            self._tracks = (b + np.tile(shift, 2), a, tg)
        if self.cfg.ground_latch:
            self.ground += float(shift[1])      # the ground line moves with the wall, so the
            self.ground_tag += float(shift[1])  # rule works on a gimbal or in the hand too
        for p in self.people:
            p.box, p.box0, p.lock0 = (x + np.tile(shift, 2) for x in (p.box, p.box0, p.lock0))
        self.shift_unknown = conf < 0.3   # blurred or unreadable: the shift above can't be trusted
        self.unstable = self.shift_unknown or float(np.linalg.norm(shift)) > self.cfg.unstable_shift * self.wh[1]

    def steady(self) -> bool:
        """A view worth learning from: camera still, climber not moving fast, no recent jump."""
        if self.unstable:
            return False
        if not self.cfg.learn_gate:
            return True
        speed = float(np.linalg.norm(self.v)) / max(self.wh[1], 1.0)
        return speed < self.cfg.learn_max_speed and self.t - self.jump_t > 1.0 and self.t - self.fall_t > 2.0

    def _common_shift(self, c: Cands) -> np.ndarray | None:
        """When the camera's own motion is unknown: if several people jumped by the same vector,
        that was the camera. Needs two people; with one, a bump and a move look alike."""
        b = self._tracks[0]
        if len(b) < 2 or len(c) < 2:
            return None
        pc, cc = (b[:, :2] + b[:, 2:]) / 2, (c.boxes[:, :2] + c.boxes[:, 2:]) / 2
        best, support = None, 1
        for i in range(len(cc)):
            for j in range(len(pc)):
                T = cc[i] - pc[j]
                n = sum(iou(m, c.boxes).max() > 0.5 for m in b + np.tile(T, 2))
                if n > support:
                    best, support = T, n
        return best

    def predicted_box(self, t: float | None = None) -> np.ndarray:
        dt = 0.0 if t is None else t - self.t
        c = self.c + self.v * dt
        return np.array([c[0] - self.wh[0] / 2, c[1] - self.wh[1] / 2, c[0] + self.wh[0] / 2, c[1] + self.wh[1] / 2])

    def step(self, t: float, view: Cands, spotter: Cands | None = None) -> Step:
        if self.tiein or (self.cfg.ground_latch and t - self.t_tag <= 3.0):
            self._observe(t, view)
        prev_state = self.state
        s = self._step(t, view, spotter)
        if self.cfg.negatives or self.cfg.transact:
            self._learn_people(t, s, prev_state, view)
        self._latch(s.state)
        if self.tiein:
            self._tiein(t, view, s)
        return s

    def _learn_people(self, t: float, s: Step, prev_state: str, view: Cands) -> None:
        """Commit or discard pending profile updates, and bank the people next to a confident lock
        as negatives."""
        cfg, p = self.cfg, self.p
        doubt = (self.unstable or self.jump_t == t
                 or (s.state == LOCKED and prev_state == LOST)
                 or (s.state == LOCKED and self.rival_gap is not None and self.rival_gap < cfg.doubt_rival))
        k = s.chosen if s.state == LOCKED and 0 <= s.chosen < len(view) else -1
        if k >= 0 and view.embs is not None and p.negs:
            one = view.subset(np.arange(len(view)) == k)
            if float(p.neg_cos(one)[0]) > float(p.pos_cos(one)[0]) + cfg.neg_margin:
                doubt = True  # the lock looks more like a known non-climber than like the climber
        if doubt:
            p.doubt()
        else:
            p.commit_due(t, cfg)
        if (not cfg.negatives or k < 0 or doubt or s.app < cfg.adapt_app or not self.steady()
                or t - p.last_neg_t < cfg.neg_every_s or view.embs is None or len(view) < 2):
            return
        _, age, _ = self._tracks
        lock = view.boxes[k]
        lock_cos = float(view.embs[k] @ p.emb) if p.emb is not None else 1.0
        same = same_person(lock, view.boxes, 0.3)
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
        for j in range(len(view)):
            if j == k or same[j] or j >= len(age) or age[j] < cfg.settle_s:
                continue
            b = view.boxes[j]
            inter = max(0.0, min(b[2], lock[2]) - max(b[0], lock[0])) * max(0.0, min(b[3], lock[3]) - max(b[1], lock[1]))
            if inter > 0.05 * min(area(b), area(lock)):
                continue  # touching the lock: too entangled to call
            if p.emb is not None and float(view.embs[j] @ p.emb) >= lock_cos:
                continue  # looks more like the tag than the lock does: a doubt, not a negative
            p.add_negative(view.embs[j], view.feats[j], view.valids[j], t, cfg)
        p.last_neg_t = t

    def _step(self, t: float, view: Cands, spotter: Cands | None = None) -> Step:
        """Advance to time t with this frame's candidates.

        `spotter` is an optional wide-camera candidate set. It competes with the recording view
        while locked/uncertain, and is the only source while lost.
        """
        dt = max(t - self.t, 1e-3)
        if self.shift_unknown and self.cfg.jump_guard:
            T = self._common_shift(view)
            if T is not None:
                self.c = self.c + T
                b, a, tg = self._tracks
                self._tracks = (b + np.tile(T, 2), a, tg)
        pred = self.c + self.v * dt
        self.t = t
        since = t - self.t_seen
        has_spotter = spotter is not None and len(spotter) > 0

        if self.state == LOST:
            c = spotter if has_spotter else view
            if len(c) == 0 or not self.allow_reacquire or self.unstable:
                self._carry(c, dt, -1)
                return self._miss(pred, since)
            s = self._reacquire(pred, c, since)
            self._carry(c, dt, s.chosen)
            return s

        # One pool: a weak match in the narrow view can't win over a strong one the wide camera sees.
        sources = np.array(["view"] * len(view) + (["spotter"] * len(spotter) if has_spotter else []))
        c = view.concat(spotter) if has_spotter else view
        if len(c) == 0:
            self._carry(c, dt, -1)
            return self._miss(pred, since)
        s = self._match(pred, since, dt, c, sources)
        self._carry(c, dt, s.chosen if s is not None else -1)
        return s if s is not None else self._miss(pred, since)

    def _track_info(self, c: Cands) -> tuple[np.ndarray, np.ndarray]:
        """Per candidate: how long it has been seen (s, by IoU with last frame) and whether that
        track is the one the lock has been following."""
        b, a, tg = self._tracks
        age, is_t = np.zeros(len(c)), np.zeros(len(c), bool)
        if len(b):
            for i, box in enumerate(c.boxes):
                ov = iou(box, b)
                j = int(np.argmax(ov))
                if ov[j] > 0.3:
                    age[i], is_t[i] = a[j], tg[j]
        return age, is_t

    def _carry(self, c: Cands, dt: float, chosen: int) -> None:
        age, is_t = self._track_info(c)
        if len(c):
            age = age + dt
            if chosen >= 0:
                is_t = np.zeros(len(c), bool)
                is_t[chosen] = True
        self._tracks = (c.boxes.copy() if len(c) else np.zeros((0, 4)), age, is_t)

    def _jumps(self, pred, since, dt, c: Cands, cents, h_ref) -> np.ndarray:
        """Candidates the lock may not move to (see Config.jump_guard)."""
        cfg = self.cfg
        hw = self.wh / 2
        gx = np.maximum.reduce([np.zeros(len(c)), c.boxes[:, 0] - (pred[0] + hw[0]), (pred[0] - hw[0]) - c.boxes[:, 2]])
        gy = np.maximum.reduce([np.zeros(len(c)), c.boxes[:, 1] - (pred[1] + hw[1]), (pred[1] - hw[1]) - c.boxes[:, 3]])
        gap = np.hypot(gx, gy) / h_ref
        apart = gap > 0
        if self.unstable:
            return apart  # camera being moved: only follow what still touches the climber
        el = max(since, dt)
        reach = cfg.move_rate * el + cfg.jump_margin
        d = (cents - pred) / h_ref
        v_down = max(float(self.v[1]), 0.0) / h_ref
        fall_reach = reach + v_down * el + 0.5 * cfg.fall_accel * el ** 2
        falling = (d[:, 1] > 0) & (gap <= fall_reach) & (np.abs(d[:, 0]) <= reach + 0.5 * d[:, 1])
        age, is_t = self._track_info(c)
        someone_else = (age >= cfg.settle_s) & ~is_t
        if cfg.others_always:
            return apart & (gap > reach) & ~falling | someone_else
        return apart & ((gap > reach) & ~falling | someone_else)

    def _geometry(self, c: Cands):
        cents = np.stack([(c.boxes[:, 0] + c.boxes[:, 2]) / 2, (c.boxes[:, 1] + c.boxes[:, 3]) / 2], 1)
        hs = np.maximum(c.boxes[:, 3] - c.boxes[:, 1], 1.0)
        return cents, hs

    def _match(self, pred, since, dt, c: Cands, sources) -> Step | None:
        cfg = self.cfg
        cents, hs = self._geometry(c)
        app = self.p.sim(c, cfg)
        known = app >= 0
        h_ref = max(self.wh[1], 30.0)
        sigma = min(0.6 + 1.5 * since, cfg.max_sigma)
        d = np.linalg.norm(cents - pred, axis=1) / h_ref
        motion = np.exp(-0.5 * (d / sigma) ** 2)
        size = np.exp(-(np.log(hs / h_ref) / 0.5) ** 2)
        floor = max(cfg.app_min, cfg.app_rel * self.app_level)
        if self.t < self.trust_until:
            floor = cfg.app_min  # referee just confirmed them: drop the relative bar, keep the basic one
        score = 0.55 * np.maximum(app, 0) + 0.30 * motion + 0.15 * size
        score -= 0.005 * (sources == "spotter")  # ties go to the recording view (more pixels)
        bad = self._implausible(c, hs, h_ref) | self._vetoed(len(c))
        if cfg.jump_guard:
            jump = self._jumps(pred, since, dt, c, cents, h_ref)
            ok = known & (app >= floor) & (motion >= 0.05) & ~bad
            if ok.any() and jump[int(np.argmax(np.where(ok, score, -1)))] and score[ok].max() >= cfg.accept:
                self.jump_t = self.t  # the guard overruled the tracker's pick: worth asking the referee
            bad |= jump
        score[~known | (app < floor) | (motion < 0.05) | bad] = 0.0
        if score.max() < cfg.accept:
            # Nobody recognisable. Someone whose appearance couldn't be judged at all is
            # acceptable only right where we expect the climber.
            fallback = 0.55 * floor + 0.30 * motion + 0.15 * size
            score = np.where(~known & (motion > 0.6) & ~bad, fallback, 0.0)

        best = int(np.argmax(score))
        if score[best] < cfg.accept:
            return None
        rivals = score[iou(c.boxes[best], c.boxes) < 0.5]  # same person seen by both cameras isn't a rival
        ambiguous = len(rivals) > 0 and rivals.max() > score[best] - 0.05
        self.rival_gap = float(score[best] - rivals.max()) if len(rivals) and rivals.max() > 0 else None
        self._update(cents[best], c.boxes[best], dt)
        if self.v[1] / h_ref > cfg.fall_speed:
            self.fall_t = self.t
        if known[best] and app[best] >= cfg.adapt_app and not ambiguous and sources[best] == "view" and self.steady():
            self.p.learn(c, best, self.t, cfg)
        if known[best]:
            self.app_level = 0.95 * self.app_level + 0.05 * float(app[best])
        self.state = LOCKED
        self.t_seen = self.t
        return Step(LOCKED, c.boxes[best].copy(), self.v.copy(), best, float(app[best]), float(score[best]),
                    str(sources[best]))

    def _vetoed(self, n: int) -> np.ndarray:
        """Referee vetoes apply to the recording-view candidates (the first len(exclude))."""
        bad = np.zeros(n, bool)
        if self.exclude is not None:
            k = min(len(self.exclude), n)
            bad[:k] = self.exclude[:k]
        return bad

    def force(self, box: np.ndarray, t: float, trust_s: float = 0.0) -> None:
        """Referee: this box is the climber. Re-lock onto it now, and for `trust_s` follow it by
        position even when the appearance match is weak (a distant climber on small models)."""
        self.trust_until = t + trust_s
        self.cont = True  # an identity anchor, like the tag
        self.c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float64)
        self.wh = np.array([box[2] - box[0], box[3] - box[1]], dtype=np.float64)
        self.v = np.zeros(2)
        self.state, self.t_seen, self._pending = LOCKED, t, None
        if self.cfg.transact:
            self.p.doubt()  # the lock was wrong: what it taught us lately is suspect
        b, a, _ = self._tracks  # the referee's pick is the target's track from now on
        tg = np.zeros(len(b), bool)
        if len(b) and iou(box, b).max() > 0.3:
            tg[int(np.argmax(iou(box, b)))] = True
        self._tracks = (b, a, tg)

    def release(self, t: float) -> None:
        """Referee: whoever we're locked onto is not the climber. Drop the lock."""
        self.state, self.t_seen, self._pending = LOST, t - 1e3, None

    def _implausible(self, c: Cands, hs: np.ndarray, h_ref: float | None) -> np.ndarray:
        """Candidates that can't be the climber whatever they look like."""
        cfg = self.cfg
        bad = np.zeros(len(c), bool)
        if h_ref is not None:
            ratio = hs / h_ref
            big = cfg.max_size_ratio
            if cfg.fall_size and self.t - self.fall_t < cfg.fall_size_s:
                big = cfg.fall_size_ratio  # falling toward the camera: they get much bigger
            bad |= (ratio > big) | (ratio < 1 / cfg.max_size_ratio)
        if cfg.ground_rule:
            feet = self.c[1] + self.wh[1] / 2
            if self.latched or self.ground - feet > 1.0 * self.h0:  # climber is well up the wall
                bad |= c.boxes[:, 3] > self.ground - 0.6 * self.h0  # people standing at the base
        return bad

    # ---- ground latch, tagged-on-the-wall check, tie-in phase (all geometry on existing boxes)

    def _latch(self, state: str) -> None:
        """Set while the lock is continuously the tagged/confirmed person and up; released only
        when the lock is followed back down (locked, touching boxes, no loss since it was up).
        Without it the rule switches off as soon as a lost lock re-acquires someone standing low."""
        if not (self.cfg.ground_rule and self.cfg.ground_latch):
            return
        box = self.predicted_box()
        prev, self._prev = self._prev, (state, box)
        up = self.ground - box[3] > 1.0 * self.h0
        if state == LOST:
            self.cont = False  # whoever is re-acquired later can't latch it; only a force() can
        if state != LOCKED:
            self._jumped_since_up |= state == LOST
            return
        followed = prev is not None and prev[0] == LOCKED and iou(prev[1], box[None])[0] > 0
        if up:
            self.latched |= self.cont
            self._up_t, self._jumped_since_up = self.t, False
        elif not followed:
            self._jumped_since_up = True
        elif self.latched and not self._jumped_since_up and self.t - self._up_t <= 5.0:
            self.latched = False

    def _observe(self, t: float, c: Cands) -> None:
        live = [i for i, p in enumerate(self.people) if t - p.t <= 1.0]
        pid = -np.ones(len(c), int)
        if live and len(c):
            M = np.array([iou(b, np.array([self.people[i].box for i in live])) for b in c.boxes])
            while M.max() >= 0.3:
                n, k = np.unravel_index(np.argmax(M), M.shape)
                p = self.people[live[k]]
                p.t, p.box = t, c.boxes[n].astype(np.float64)
                pid[n] = live[k]
                M[n, :] = M[:, k] = -1
        for n in np.where(pid < 0)[0]:
            b = c.boxes[n].astype(np.float64)
            self.people.append(_Person(t, t, b, b.copy(), self.predicted_box()))
            pid[n] = len(self.people) - 1
        self._pid = pid
        if self.cfg.ground_latch and t - self.t_tag <= 3.0:
            self._tagged_on_wall(c)

    def _tagged_on_wall(self, c: Cands) -> None:
        """Someone beside the tag, at least 0.8x as tall, with their *head* below the tag's feet:
        the tag isn't standing on the floor (with the camera below head height, a standing
        person's feet are always below any standing person's head). Their feet are the ground,
        and there is no tie-in to watch. (IMG_2891: tagged mid-wall, the lock fell to the base.)"""
        tb = self.predicted_box()
        for b in c.boxes:
            gap = max(0.0, b[0] - tb[2], tb[0] - b[2]) / max(tb[2] - tb[0], 1.0)
            if gap <= 1.0 and b[1] >= tb[3] and b[3] - b[1] >= 0.8 * (tb[3] - tb[1]):
                g = float(self.frame_h) if self.frame_h is not None and b[3] >= self.frame_h - 3 else float(b[3])
                self.ground = g if self.ground == self.ground_tag else min(self.ground, g)
                self.tiein = False

    def _tiein(self, t: float, c: Cands, s: Step) -> None:
        cfg = self.cfg
        lb = self.predicted_box()
        hl, lw = max(lb[3] - lb[1], 1.0), max(lb[2] - lb[0], 1.0)
        if t - self.t_tag > cfg.tiein_s or (self.cont and s.state == LOCKED and self.ground_tag - lb[3] > 2.0 * hl):
            self.tiein, self.people = False, []  # timed out, or the lock itself left the ground
            return
        app = self.tag_profile.sim(c, cfg) if len(c) else np.zeros(0)
        for n, i in enumerate(self._pid):
            if app[n] >= 0:
                self.people[i].apps = (self.people[i].apps + [float(app[n])])[-10:]
        lock_i = int(self._pid[s.chosen]) if 0 <= s.chosen < len(self._pid) else -1
        if lock_i >= 0 and app[s.chosen] >= 0:  # (view candidates come first; spotter ones aren't tracked)
            self._lock_apps = (self._lock_apps + [float(app[s.chosen])])[-10:]
        if self.ground_tag - lb[3] > 0.5 * hl:
            return  # the lock isn't standing where the climber tied in
        risers = []
        for n, i in enumerate(self._pid):
            p = self.people[i]
            if i == lock_i or p.t0 < self.t_tag:
                continue
            b, b0, L, h = p.box, p.box0, p.lock0, p.box[3] - p.box[1]
            born_beside = max(0.0, b0[0] - L[2], L[0] - b0[2]) <= max(L[2] - L[0], 1.0)  # appeared beside the lock
            up = (b[3] <= lb[3] - cfg.riser_feet_h * h and b[1] <= lb[1] - cfg.riser_head_h * h
                  and h >= cfg.riser_size * hl and abs((b[0] + b[2]) / 2 - (lb[0] + lb[2]) / 2) <= cfg.riser_column * lw)
            if up and born_beside:
                p.up_since = t if p.up_since is None else p.up_since
                if t - p.up_since >= cfg.riser_hold_s:
                    risers.append(p)
            else:
                p.up_since = None
        la = float(np.median(self._lock_apps)) if self._lock_apps else 0.0
        ok = [p for p in risers if p.apps and np.median(p.apps) >= max(cfg.riser_floor, la + cfg.riser_margin)]
        if len(ok) != 1:
            return  # nobody, or ambiguous: leave it to the referee
        r = ok[0]
        self.events.append((t, "ascent swap", r.box.copy()))
        lock_feet = lb[3]
        self.force(r.box, t)
        self.jump_t = t                           # the referee, if any, checks the swap
        self.p = copy.deepcopy(self.tag_profile)  # forget what was learned from the wrong person
        self.app_level = 0.85
        self.ground = max(self.ground, lock_feet)   # where the lock stood is the ground...
        self.h0 = float(r.box[3] - r.box[1])        # ...in the climber's units
        self.tiein, self.people = False, []

    def _update(self, c: np.ndarray, box: np.ndarray, dt: float) -> None:
        pred = self.c + self.v * dt
        r = c - pred
        self.c = pred + 0.6 * r  # alpha-beta filter
        self.v = self.v + 0.15 * r / dt
        self.wh = 0.7 * self.wh + 0.3 * np.array([box[2] - box[0], box[3] - box[1]])

    def _miss(self, pred: np.ndarray, since: float) -> Step:
        self.c = pred
        self.v *= 0.8
        self.state = UNCERTAIN if since < self.cfg.uncertain_s else LOST
        return Step(self.state, self.predicted_box(), self.v.copy(), -1, 0.0, 0.0)

    def _reacquire(self, pred, c: Cands, since) -> Step:
        cfg = self.cfg
        cents, hs = self._geometry(c)
        app = self.p.sim(c, cfg)
        app[self._implausible(c, hs, None) | self._vetoed(len(c))] = -1.0
        best = int(np.argmax(app))
        if app[best] < cfg.reacquire_app:
            self._pending = None
            return self._miss(pred, since)
        if self._pending is not None and np.linalg.norm(cents[best] - self._pending[0]) < 1.0 * hs[best]:
            self._pending = (cents[best], self._pending[1] + 1)
        else:
            self._pending = (cents[best], 1)
        if self._pending[1] < cfg.reacquire_frames:
            return self._miss(pred, since)
        self._pending = None
        b = c.boxes[best]
        self.c = cents[best].astype(np.float64)
        self.v = np.zeros(2)
        self.wh = np.array([b[2] - b[0], b[3] - b[1]], dtype=np.float64)
        self.state = LOCKED
        self.t_seen = self.t
        return Step(LOCKED, b.copy(), self.v.copy(), best, float(app[best]), float(app[best]), "spotter")


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of box a [4] against boxes b [N,4]."""
    x0 = np.maximum(a[0], b[:, 0]); y0 = np.maximum(a[1], b[:, 1])
    x1 = np.minimum(a[2], b[:, 2]); y1 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])  # noqa: E731
    return inter / np.maximum(area(a) + area(b) - inter, 1e-6)


def same_person(a: np.ndarray, b: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Box a [4] vs boxes b [N,4]: True where they're probably the same person. Overlap by IoU,
    or one box mostly inside the other and at least a quarter of its height (a detector that
    boxed only the upper body). A distant person inside a near person's box is far smaller, so
    it stays separate."""
    if not len(b):
        return np.zeros(0, bool)
    x0 = np.maximum(a[0], b[:, 0]); y0 = np.maximum(a[1], b[:, 1])
    x1 = np.minimum(a[2], b[:, 2]); y1 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda x: np.maximum((x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1]), 1e-6)  # noqa: E731
    contained = inter / np.minimum(area(a), area(b))
    ha, hb = a[3] - a[1], b[:, 3] - b[:, 1]
    similar = np.minimum(ha, hb) >= 0.25 * np.maximum(ha, hb)
    return (iou(a, b) >= iou_thr) | ((contained >= 0.7) & similar)


def register(clip, dets) -> tuple[Profile, np.ndarray, int]:
    """Tag the clip's target the way the clip says: an operator's tap, or the L pose."""
    if clip.register == "pose":
        from . import registration
        events, _ = registration.scan(dets)
        if not events:
            raise SystemExit(f"{clip.id}: nobody held the L pose; nothing to register")
        e = events[0]
        W, H = dets.size
        b = e.box
        seed = (e.t, float((b[0] + b[2]) / 2 / W), float((b[1] + b[3]) / 2 / H))
        print(f"[register] {clip.id}: L pose at {e.t:.1f}s")
        # Keep learning while they turn to the wall and tie in: the back is what we'll film.
        return tag(dets, seed, seconds=6.0, max_seconds=8.0)
    return tag(dets, clip.seed)


def tag(dets, seed: tuple[float, float, float], seconds: float = 1.5, max_seconds: float = 5.0
        ) -> tuple[Profile, np.ndarray, int]:
    """The "tag climber" moment: pick the detection under the tap, follow it by overlap for a
    moment (longer if the body wasn't seen well enough), and average its appearance.
    Returns (profile, box, frame index)."""
    t_seed, fx, fy = seed
    W, H = dets.size
    i0 = int(np.argmin(np.abs(dets.t - t_seed)))
    for i in range(i0, len(dets)):
        sl = dets.frame(i)
        b = dets.boxes[sl]
        if len(b) == 0:
            continue
        px, py = fx * W, fy * H
        inside = (b[:, 0] <= px) & (px <= b[:, 2]) & (b[:, 1] <= py) & (py <= b[:, 3])
        cents = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
        dist = np.linalg.norm(cents - [px, py], axis=1) - 1e6 * inside
        k = int(np.argmin(dist))
        if inside[k] or dist[k] < 0.1 * H:
            break
    else:
        raise SystemExit(f"no detection near seed {seed}")
    start = i
    first = sl.start + k
    box = dets.boxes[first]
    picked = [first]
    j = i

    def evidence() -> float:
        seen = dets.fvalid[picked].any(axis=0)
        return float(appearance.WEIGHTS[seen].sum())

    while j + 1 < len(dets) and (dets.t[j + 1] - dets.t[start] <= seconds or
                                 (evidence() < 0.85 and dets.t[j + 1] - dets.t[start] <= max_seconds)):
        j += 1
        sl = dets.frame(j)
        if len(dets.boxes[sl]) == 0:
            continue
        ov = iou(box, dets.boxes[sl])
        m = int(np.argmax(ov))
        if ov[m] > 0.4:
            box = dets.boxes[sl][m]
            picked.append(sl.start + m)
    F, V = dets.feat[picked], dets.fvalid[picked]
    cnt = V.sum(axis=0)
    mean = (F * V[..., None]).sum(axis=0) / np.maximum(cnt, 1)[:, None]
    emb = None
    embs = getattr(dets, EMB_KEY, None)
    if embs is not None:
        e = embs[picked].mean(axis=0)
        emb = (e / max(np.linalg.norm(e), 1e-9)).astype(np.float32)
    return Profile(mean.astype(np.float32), cnt > 0, emb), dets.boxes[first].copy(), start


def selftest() -> None:
    """Jump guard on synthetic scenes where everyone looks identical, so only geometry decides:
    a long fall at 10 and 5 fps must be followed; a lookalike next to an occluded climber must
    not take the lock; a camera bump must not, whether or not the shift is known."""
    H = 200.0  # body height, px

    def cands(centres) -> Cands:
        n = len(centres)
        b = np.array([[x - 0.3 * H, y - H / 2, x + 0.3 * H, y + H / 2] for x, y in centres]).reshape(-1, 4)
        f = np.full((n, len(appearance.REGIONS), appearance.DIM), 1.0 / appearance.DIM, np.float32)
        return Cands(b, f, np.ones((n, len(appearance.REGIONS)), bool))

    def tracker(cfg=None) -> TargetTracker:
        c = cands([(500, 600)])
        prof = Profile(c.feats[0].copy(), c.valids[0].copy())
        return TargetTracker(prof, c.boxes[0], 0.0, cfg)

    def on(step, x, y) -> bool:
        return step.state == LOCKED and abs((step.box[0] + step.box[2]) / 2 - x) < 0.3 * H and \
            abs((step.box[1] + step.box[3]) / 2 - y) < 0.3 * H

    g = 9.8 / 1.7 * H  # px/s^2 for a 1.7 m climber
    for fps in (10, 5):
        trk, dt = tracker(Config(ground_rule=False)), 1.0 / fps
        for k in range(1, int(3 * fps)):
            t = k * dt
            tf = min(max(t - 1.0, 0.0), 1.2)  # 1.2 s free fall (~7 m) from t = 1 s, then caught
            y = 600 + 0.5 * g * tf ** 2
            s = trk.step(t, cands([(500, y), (500 - 3 * H, 600 + 3 * H)]))  # + a bystander below
            assert on(s, 500, y), f"lost the falling climber at {fps} fps, t={t:.1f}: {s.state} {s.box}"

    def lookalike(cfg) -> bool:
        """Climber hidden for 1.5 s; an identical person climbs 1.3 body heights to the side."""
        trk = tracker(cfg)
        for k in range(1, 40):
            t = k * 0.1
            hidden = 1.0 <= t < 2.5
            s = trk.step(t, cands(([] if hidden else [(500, 600)]) + [(500 - 1.3 * H, 600)]))
            if on(s, 500 - 1.3 * H, 600):
                return True
        assert on(s, 500, 600), f"didn't get the climber back: {s.state} {s.box}"
        return False

    assert lookalike(Config(jump_guard=False)), "test is too easy: the unguarded tracker didn't switch"
    assert not lookalike(Config()), "a lookalike took the lock while the climber was hidden"

    for known in (True, False):
        trk = tracker()
        off = np.zeros(2)
        for k in range(1, 40):
            t = k * 0.1
            shift = np.array([300.0, -250.0]) if k == 15 else np.zeros(2)  # someone knocks the phone
            off += shift
            trk.camera(shift if known else np.zeros(2), 0.9 if known or k != 15 else 0.0)
            s = trk.step(t, cands([(500 + off[0], 600 + off[1]), (500 - 1.3 * H + off[0], 600 + off[1])]))
            assert not on(s, 500 - 1.3 * H + off[0], 600 + off[1]), f"bump (known={known}) gave the lock away at t={t:.1f}"
            if known:
                assert on(s, 500 + off[0], 600 + off[1]), f"known bump lost the climber at t={t:.1f}: {s.state}"

    def tiein(slid: bool, tiein_s: float = 60.0) -> bool:
        """Tie-in: climber A (colour 0) beside belayer B (mostly colour 5). If the lock slid onto
        B while A was hidden, A climbing up beside B must take it back. If the lock stayed on A
        (a queue: B starts up the same line), it must not move."""
        def people(pts, cols):
            c = cands(pts)
            for i, col in enumerate(cols):
                c.feats[i] = 0.0
                for k, w in col:
                    c.feats[i, :, k] = w
            return c
        A, B = [(0, 1.0)], [(0, 0.6), (5, 0.4)]  # B: a lookalike to the small model (~0.77)
        tag = people([(500, 600)], [A])
        trk = TargetTracker(Profile(tag.feats[0].copy(), tag.valids[0].copy()), tag.boxes[0], 0.0,
                            Config(tiein_s=tiein_s))
        a, b = (500.0, 600.0), (660.0, 600.0)
        for k in range(1, 80):
            t = k * 0.1
            if slid and abs(t - 1.0) < 1e-6:
                trk.force(people([b], [B]).boxes[0], t)  # the lock slides onto the belayer
            if t >= 2.0:
                a = (560.0, 600.0 - 1.0 * H * (t - 2.0)) if slid else a
                b = b if slid else (660.0, 600.0 - 1.0 * H * (t - 2.0))  # queue: B climbs
            hidden = slid and 1.0 <= t < 2.0
            pts, cols = ([] if hidden else [a]) + [b], ([] if hidden else [A]) + [B]
            s = trk.step(t, people(pts, cols))
        return on(s, *a)

    assert not tiein(slid=True, tiein_s=0.0), "test is too easy: without the tie-in phase the lock came back anyway"
    assert tiein(slid=True), "the climber left the ground beside the belayer, but the lock stayed"
    assert tiein(slid=False), "queue: someone else started up beside the climber and took the lock"
    print("tracker selftest ok")


if __name__ == "__main__":
    selftest()
