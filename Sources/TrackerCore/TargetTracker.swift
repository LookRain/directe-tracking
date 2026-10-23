import Foundation

/// What the tracker says at one tick (lab tracker `Step`).
public struct Step: Sendable {
    public enum Source: String, Sendable { case view, spotter }

    public var state: TrackState
    /// The chosen candidate's box while locked, else the predicted box.
    public var box: Box
    /// px/s.
    public var velocity: SIMD2<Double>
    /// Index into this tick's candidates (view first, then spotter), -1 if none.
    public var chosen: Int
    public var app: Double
    public var score: Double
    public var source: Source = .view

    public func point(at t: Double) -> TrackPoint { TrackPoint(t: t, state: state, box: box) }
}

/// Something the tracker decided on its own (lab `events`), e.g. "ascent swap".
public struct TrackerEvent: Sendable {
    public var t: Double
    public var kind: String
    public var box: Box
}

/// Keeps one tagged climber across detections, with locked/uncertain/lost states: a faithful port of
/// directe_tracking/tracker.py `TargetTracker` (numerics included, see Numerics.swift; golden-tested bit-exact).
///
/// Driving it (as directe_tracking/online.py `run`; coordinates are upright source pixels):
///   - Tag: from the tap time, collect one tick per sample (10 fps) of `Perceiver.perceive` around
///     the tap until a tick is >= `tagWindowSeconds` after the first (that one included), then
///     `TargetTracker.start(window:point:frameHeight:config:)` (tag + init + step every window tick,
///     no camera calls). Feed CameraMotion every frame from the first window frame on, so the first
///     live shift is relative to the last window frame.
///   - Then per tick, in this order: `camera(shift:conf:)` (the camera's own motion since the last
///     tick) -> `predictedBox(at: t)` (where to look: the focus window) -> Perception -> (referee
///     only) `exclude` / `allowReacquire` / `force` / `release` -> `step(t:view:)`.
///   - Config: a preset (`.combo`, `.comboPlain`) or `TrackerConfig()`. The lab turns the ground
///     rule off for handheld footage unless the runner option ground_handheld is set (the combo
///     variants set it, and `camera` then moves the ground line with the wall): do the same.
///   - Candidate boxes are used as float32 (like the lab's); `force` boxes as float64.
public final class TargetTracker {
    public let config: TrackerConfig
    /// The live profile (replaced by the tag-time one after an ascent swap).
    public private(set) var profile: Profile
    public private(set) var state: TrackState = .locked
    /// Time of the last step.
    public private(set) var t: Double
    /// Score margin over the next-best person at the last match (nil: no rival).
    public private(set) var rivalGap: Double?
    /// Last time the jump guard refused a move (worth asking a referee).
    public private(set) var jumpT = -1e9
    /// Per-candidate veto for the next step (referee); applies to the first `exclude.count` view candidates.
    public var exclude: [Bool]?
    /// The referee turns this off: it decides re-locks.
    public var allowReacquire = true
    /// Camera being bumped/handled (last `camera` call): no identity changes.
    public private(set) var unstable = false
    public private(set) var shiftUnknown = false
    /// Ground rule latch: "the climber is up".
    public private(set) var latched = false
    /// Tie-in phase still watching.
    public private(set) var tiein: Bool
    /// Ground line (feet at the tag, moved with the camera under `groundLatch`), source px.
    public private(set) var ground: Double
    /// Running average of the target's appearance match.
    public private(set) var appLevel = 0.85
    public private(set) var events: [TrackerEvent] = []
    public let frameHeight: Double?

    var c: SIMD2<Double>
    var wh: SIMD2<Double>
    var v = SIMD2<Double>(0, 0)
    var tSeen: Double
    /// numpy float64 scalar (strong) rather than a Python float: decides the precision of the ground
    /// rule's comparison with the float32 boxes.
    var groundStrong = false
    var h0: Double
    var reacq: (cent: SIMD2<Double>, n: Int)?
    var trustUntil = -1e9
    var fallT = -1e9
    var tracks = Tracks()
    var tTag: Double
    var groundTag: Double
    let tagProfile: Profile
    var people: [Person] = []
    var pid: [Int] = []
    var lockApps: [Double] = []
    var cont = true
    var upT = -1e9
    var jumpedSinceUp = false
    var prev: (state: TrackState, box: Box)?

    /// Last frame's boxes (as the lab's `_tracks`): age (s) and whether that track is the target.
    struct Tracks {
        var boxes: [Box] = []
        var f32 = false
        var age: [Double] = []
        var isT: [Bool] = []
    }

    /// Short-lived IoU track of anyone (tie-in phase and the tagged-on-the-wall check).
    struct Person {
        var t0: Double
        var t: Double
        var box: Box
        var box0: Box
        var lock0: Box
        var upSince: Double?
        var apps: [Double] = []
    }

    /// `box`: the tagged candidate's box (from `tag`), `t`: its time, `frameHeight`: source frame
    /// height in px (the ground latch needs it).
    public init(profile: Profile, box: Box, t: Double, config: TrackerConfig = TrackerConfig(), frameHeight: Double? = nil) {
        let b = box.float32
        self.profile = profile
        self.config = config
        c = SIMD2(f32(b.x0 + b.x1) / 2, f32(b.y0 + b.y1) / 2)
        wh = SIMD2(f32(b.x1 - b.x0), f32(b.y1 - b.y0))
        self.t = t
        tSeen = t
        ground = b.y1  // feet at the tag moment: ground level
        h0 = f32(b.y1 - b.y0)
        if config.groundLatch, let fh = frameHeight, b.y1 >= fh - 3 {
            ground = fh  // feet below the frame: the ground is at least the bottom edge
        }
        self.frameHeight = frameHeight
        tTag = t
        groundTag = ground
        tagProfile = profile  // the tie-in compares against this, never the live profile
        tiein = config.tieinS > 0
    }

    // MARK: camera

    /// The camera itself moved the image by `shift` px since the last frame (gyro / image
    /// registration). Moves what we know with it, so people's motion is judged against the wall.
    /// `conf` < 0.3 (blurred, unreadable) or a move over `unstableShift` body heights freezes identity
    /// changes until it settles.
    public func camera(shift: SIMD2<Double>, conf: Double = 1.0) {
        c = c + shift
        if !tracks.boxes.isEmpty {
            tracks.boxes = tracks.boxes.map { $0.offset(shift) }
            tracks.f32 = false
        }
        if config.groundLatch {
            ground += shift.y  // the ground line moves with the wall
            groundTag += shift.y
        }
        for i in people.indices {
            people[i].box = people[i].box.offset(shift)
            people[i].box0 = people[i].box0.offset(shift)
            people[i].lock0 = people[i].lock0.offset(shift)
        }
        shiftUnknown = conf < 0.3
        unstable = shiftUnknown || norm(shift) > config.unstableShift * wh.y
    }

    /// A view worth learning from: camera still, climber not moving fast, no recent jump.
    public func steady() -> Bool {
        if unstable { return false }
        if !config.learnGate { return true }
        let speed = norm(v) / max(wh.y, 1.0)
        return speed < config.learnMaxSpeed && t - jumpT > 1.0 && t - fallT > 2.0
    }

    /// When the camera's own motion is unknown: if several people jumped by the same vector, that
    /// was the camera. Needs two people; with one, a bump and a move look alike.
    func commonShift(_ cc: Cands) -> (v: SIMD2<Double>, f32: Bool)? {
        let b = tracks.boxes, bf = tracks.f32
        guard b.count >= 2, cc.count >= 2 else { return nil }
        @inline(__always) func r(_ x: Double) -> Double { bf ? f32(x) : x }
        let pc = b.map { SIMD2(r($0.x0 + $0.x1) / 2, r($0.y0 + $0.y1) / 2) }
        let cents = cc.geometry().cents
        var best: SIMD2<Double>?
        var support = 1
        for i in cents.indices {
            for j in pc.indices {
                let T = SIMD2(r(cents[i].x - pc[j].x), r(cents[i].y - pc[j].y))
                var n = 0
                for m in b {
                    let moved = Box(r(m.x0 + T.x), r(m.y0 + T.y), r(m.x1 + T.x), r(m.y1 + T.y))
                    let ov = npIoU(moved, bf, cc.boxes, true)
                    if gt(ov.v.max()!, 0.5, f32: ov.f32) { n += 1 }
                }
                if n > support { best = T; support = n }
            }
        }
        return best.map { ($0, bf) }
    }

    /// Where the climber is expected at `t` (now if nil).
    public func predictedBox(at t: Double? = nil) -> Box {
        let dt = t.map { $0 - self.t } ?? 0.0
        let p = c + v * dt
        return Box(p.x - wh.x / 2, p.y - wh.y / 2, p.x + wh.x / 2, p.y + wh.y / 2)
    }

    // MARK: step

    /// Advance to time `t` with this frame's candidates. `spotter`: an optional wide-camera set that
    /// competes with the recording view while locked/uncertain and is the only source while lost.
    public func step(t: Double, view: [Candidate], spotter: [Candidate]? = nil) -> Step {
        step(t, Cands(view), spotter.map { Cands($0) })
    }

    func step(_ t: Double, _ view: Cands, _ spotter: Cands? = nil) -> Step {
        if tiein || (config.groundLatch && t - tTag <= 3.0) { observe(t, view) }
        let prevState = state
        let s = innerStep(t, view, spotter)
        if config.negatives || config.transact { learnPeople(t, s, prevState, view) }
        latch(s.state)
        if tiein { tieIn(t, view, s) }
        return s
    }

    /// Commit or discard pending profile updates, and bank the people next to a confident lock as
    /// negatives.
    func learnPeople(_ t: Double, _ s: Step, _ prevState: TrackState, _ view: Cands) {
        let cfg = config
        var doubt = unstable || jumpT == t || (s.state == .locked && prevState == .lost)
            || (s.state == .locked && rivalGap != nil && rivalGap! < cfg.doubtRival)
        let k = s.state == .locked && s.chosen >= 0 && s.chosen < view.count ? s.chosen : -1
        if k >= 0, view.embs != nil, !profile.negs.isEmpty {
            let one = view.subset(k)
            if let neg = profile.negCos(one), let pos = profile.posCos(one), Double(neg[0]) > Double(pos[0]) + cfg.negMargin {
                doubt = true  // the lock looks more like a known non-climber than like the climber
            }
        }
        if doubt {
            profile.doubt()
        } else {
            profile.commitDue(t, cfg)
        }
        guard cfg.negatives, k >= 0, !doubt, !(s.app < cfg.adaptApp), steady(),
              !(t - profile.lastNegT < cfg.negEveryS), view.embs != nil, view.count >= 2 else { return }
        let age = tracks.age
        let lock = view.boxes[k]
        let lockCos = profile.emb.map { Double(dot(view.emb(k), $0)) } ?? 1.0
        let same = npSamePerson32(lock, view.boxes, 0.3)
        let lockArea = npArea(lock, f32: true)
        for j in 0..<view.count {
            if j == k || same[j] || j >= age.count || age[j] < cfg.settleS { continue }
            let b = view.boxes[j]
            let iw = max(0.0, f32(min(b.x1, lock.x1) - max(b.x0, lock.x0)))
            let ih = max(0.0, f32(min(b.y1, lock.y1) - max(b.y0, lock.y0)))
            if f32(iw * ih) > f32(weak32(0.05) * min(npArea(b, f32: true), lockArea)) {
                continue  // touching the lock: too entangled to call
            }
            if let anchor = profile.emb, Double(dot(view.emb(j), anchor)) >= lockCos {
                continue  // looks more like the tag than the lock does: a doubt, not a negative
            }
            profile.addNegative(Array(view.emb(j)), view.feat(j), view.valid(j), t, cfg)
        }
        profile.lastNegT = t
    }

    func innerStep(_ t: Double, _ view: Cands, _ spotter: Cands?) -> Step {
        let dt = max(t - self.t, 1e-3)
        if shiftUnknown && config.jumpGuard, let T = commonShift(view) {
            c = c + T.v
            let p = tracks.f32 && T.f32
            @inline(__always) func r(_ x: Double) -> Double { p ? f32(x) : x }
            tracks.boxes = tracks.boxes.map { Box(r($0.x0 + T.v.x), r($0.y0 + T.v.y), r($0.x1 + T.v.x), r($0.y1 + T.v.y)) }
            tracks.f32 = p
        }
        let pred = c + v * dt
        self.t = t
        let since = t - tSeen
        let hasSpotter = (spotter?.count ?? 0) > 0

        if state == .lost {
            let cc = hasSpotter ? spotter! : view
            if cc.count == 0 || !allowReacquire || unstable {
                carry(cc, dt, -1)
                return miss(pred, since)
            }
            let s = reacquire(pred, cc, since)
            carry(cc, dt, s.chosen)
            return s
        }

        // One pool: a weak match in the narrow view can't win over a strong one the wide camera sees.
        var sources = [Step.Source](repeating: .view, count: view.count)
        var cc = view
        if hasSpotter {
            sources += [Step.Source](repeating: .spotter, count: spotter!.count)
            cc = view.concat(spotter!)
        }
        if cc.count == 0 {
            carry(cc, dt, -1)
            return miss(pred, since)
        }
        let s = match(pred, since, dt, cc, sources)
        carry(cc, dt, s?.chosen ?? -1)
        return s ?? miss(pred, since)
    }

    /// Per candidate: how long it has been seen (s, by IoU with last frame) and whether that track is
    /// the one the lock has been following.
    func trackInfo(_ cc: Cands) -> (age: [Double], isT: [Bool]) {
        var age = [Double](repeating: 0, count: cc.count), isT = [Bool](repeating: false, count: cc.count)
        if !tracks.boxes.isEmpty {
            for (i, box) in cc.boxes.enumerated() {
                let ov = npIoU(box, true, tracks.boxes, tracks.f32)
                let j = argmax(ov.v)
                if gt(ov.v[j], 0.3, f32: ov.f32) {
                    age[i] = tracks.age[j]
                    isT[i] = tracks.isT[j]
                }
            }
        }
        return (age, isT)
    }

    func carry(_ cc: Cands, _ dt: Double, _ chosen: Int) {
        var (age, isT) = trackInfo(cc)
        if cc.count > 0 {
            age = age.map { $0 + dt }
            if chosen >= 0 {
                isT = [Bool](repeating: false, count: cc.count)
                isT[chosen] = true
            }
        }
        tracks = Tracks(boxes: cc.boxes, f32: cc.count > 0, age: age, isT: isT)
    }

    /// Candidates the lock may not move to (see TrackerConfig.jumpGuard).
    func jumps(_ pred: SIMD2<Double>, _ since: Double, _ dt: Double, _ cc: Cands, _ cents: [SIMD2<Double>], _ hRef: Double) -> [Bool] {
        let cfg = config
        let hw = wh / 2
        var gap = [Double](), apart = [Bool]()
        for b in cc.boxes {
            let gx = max(max(0, b.x0 - (pred.x + hw.x)), (pred.x - hw.x) - b.x1)
            let gy = max(max(0, b.y0 - (pred.y + hw.y)), (pred.y - hw.y) - b.y1)
            let g = hypot(gx, gy) / hRef
            gap.append(g)
            apart.append(g > 0)
        }
        if unstable { return apart }  // camera being moved: only follow what still touches the climber
        let el = max(since, dt)
        let reach = cfg.moveRate * el + cfg.jumpMargin
        let vDown = max(v.y, 0.0) / hRef
        let fallReach = reach + vDown * el + 0.5 * cfg.fallAccel * (el * el)
        let (age, isT) = trackInfo(cc)
        return cc.boxes.indices.map { i in
            let dx = (cents[i].x - pred.x) / hRef, dy = (cents[i].y - pred.y) / hRef
            let falling = dy > 0 && gap[i] <= fallReach && abs(dx) <= reach + 0.5 * dy
            let someoneElse = age[i] >= cfg.settleS && !isT[i]
            if cfg.othersAlways { return (apart[i] && gap[i] > reach && !falling) || someoneElse }
            return apart[i] && ((gap[i] > reach && !falling) || someoneElse)
        }
    }

    func match(_ pred: SIMD2<Double>, _ since: Double, _ dt: Double, _ cc: Cands, _ sources: [Step.Source]) -> Step? {
        let cfg = config
        let n = cc.count
        let (cents, hs) = cc.geometry()
        let app = profile.sim(cc, cfg)
        let known = app.v.map { $0 >= 0 }
        // max(wh[1], 30.0) is the (weakly typed) Python float 30.0 only when wh[1] < 30: then
        // hs / h_ref runs in float32
        let hRefWeak = wh.y < 30.0
        let hRef = max(wh.y, 30.0)
        let sigma = min(0.6 + 1.5 * since, cfg.maxSigma)
        var motion = [Double](repeating: 0, count: n), size = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let dx = cents[i].x - pred.x, dy = cents[i].y - pred.y
            let q = (0 + dx * dx + dy * dy).squareRoot() / hRef / sigma
            motion[i] = exp(-0.5 * (q * q))
            if hRefWeak {
                let l = log(Float(hs[i]) / Float(30.0 as Double)) / Float(0.5 as Double)
                size[i] = Double(exp(-(l * l)))
            } else {
                let l = log(hs[i] / hRef) / 0.5
                size[i] = exp(-(l * l))
            }
        }
        @inline(__always) func sizeTerm(_ i: Int) -> Double {
            hRefWeak ? Double(Float(0.15 as Double) * Float(size[i])) : 0.15 * size[i]
        }
        var floor = max(cfg.appMin, cfg.appRel * appLevel)
        if t < trustUntil { floor = cfg.appMin }  // referee just confirmed them: drop the relative bar
        var score = (0..<n).map { i -> Double in
            let a = max(app.v[i], 0)
            let appTerm = app.f32 ? Double(Float(0.55 as Double) * Float(a)) : 0.55 * a
            return appTerm + 0.30 * motion[i] + sizeTerm(i) - (sources[i] == .spotter ? 0.005 : 0.0)
        }
        let vetoes = vetoed(n)
        var bad = implausible(cc, hs, hRef, hRefWeak).enumerated().map { $1 || vetoes[$0] }
        if cfg.jumpGuard {
            let jump = jumps(pred, since, dt, cc, cents, hRef)
            let ok = (0..<n).map { known[$0] && app.ge($0, floor) && motion[$0] >= 0.05 && !bad[$0] }
            if ok.contains(true) {
                let pick = argmax((0..<n).map { ok[$0] ? score[$0] : -1 })
                let okMax = (0..<n).filter { ok[$0] }.map { score[$0] }.max()!
                if jump[pick] && okMax >= cfg.accept {
                    jumpT = t  // the guard overruled the tracker's pick: worth asking the referee
                }
            }
            for i in 0..<n where jump[i] { bad[i] = true }
        }
        for i in 0..<n where !known[i] || app.lt(i, floor) || motion[i] < 0.05 || bad[i] { score[i] = 0 }
        if score.max()! < cfg.accept {
            // Nobody recognisable. Someone whose appearance couldn't be judged at all is acceptable
            // only right where we expect the climber.
            score = (0..<n).map { i in
                !known[i] && motion[i] > 0.6 && !bad[i] ? 0.55 * floor + 0.30 * motion[i] + sizeTerm(i) : 0.0
            }
        }
        let best = argmax(score)
        if score[best] < cfg.accept { return nil }
        let ov = npIoU(cc.boxes[best], true, cc.boxes, true)
        let rivals = (0..<n).filter { ov.v[$0] < weak32(0.5) }.map { score[$0] }  // same person seen twice isn't a rival
        let rmax = rivals.max()
        let ambiguous = rmax.map { $0 > score[best] - 0.05 } ?? false
        rivalGap = rmax.flatMap { $0 > 0 ? score[best] - $0 : nil }
        update(cents[best], cc.boxes[best], dt)
        if v.y / hRef > cfg.fallSpeed { fallT = t }
        if known[best] && app.ge(best, cfg.adaptApp) && !ambiguous && sources[best] == .view && steady() {
            profile.learn(cc, best, t, cfg)
        }
        if known[best] { appLevel = 0.95 * appLevel + 0.05 * app.v[best] }
        state = .locked
        tSeen = t
        return Step(state: .locked, box: cc.boxes[best], velocity: v, chosen: best, app: app.v[best],
                    score: score[best], source: sources[best])
    }

    /// Referee vetoes apply to the recording-view candidates (the first `exclude.count`).
    func vetoed(_ n: Int) -> [Bool] {
        var bad = [Bool](repeating: false, count: n)
        if let ex = exclude { for i in 0..<min(ex.count, n) { bad[i] = ex[i] } }
        return bad
    }

    /// Referee: this box is the climber. Re-lock onto it now, and for `trustS` follow it by position
    /// even when the appearance match is weak.
    public func force(box: Box, t: Double, trustS: Double = 0) {
        trustUntil = t + trustS
        cont = true  // an identity anchor, like the tag
        c = SIMD2((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
        wh = SIMD2(box.x1 - box.x0, box.y1 - box.y0)
        v = .zero
        state = .locked
        tSeen = t
        reacq = nil
        if config.transact { profile.doubt() }  // the lock was wrong: what it taught us lately is suspect
        var tg = [Bool](repeating: false, count: tracks.boxes.count)  // the pick is the target's track from now on
        if !tracks.boxes.isEmpty {
            let ov = npIoU(box, false, tracks.boxes, tracks.f32)
            if gt(ov.v.max()!, 0.3, f32: ov.f32) { tg[argmax(ov.v)] = true }
        }
        tracks.isT = tg
    }

    /// Referee: whoever we're locked onto is not the climber. Drop the lock.
    public func release(t: Double) {
        state = .lost
        tSeen = t - 1e3
        reacq = nil
    }

    /// Candidates that can't be the climber whatever they look like.
    func implausible(_ cc: Cands, _ hs: [Double], _ hRef: Double?, _ hRefWeak: Bool) -> [Bool] {
        let cfg = config
        var bad = [Bool](repeating: false, count: cc.count)
        if let hRef {
            var big = cfg.maxSizeRatio
            if cfg.fallSize && t - fallT < cfg.fallSizeS { big = cfg.fallSizeRatio }  // falling toward the camera
            let small = 1 / cfg.maxSizeRatio
            for i in 0..<cc.count {
                if hRefWeak {
                    let ratio = Float(hs[i]) / Float(hRef)
                    bad[i] = ratio > Float(big) || ratio < Float(small)
                } else {
                    let ratio = hs[i] / hRef
                    bad[i] = ratio > big || ratio < small
                }
            }
        }
        if cfg.groundRule {
            let feet = c.y + wh.y / 2
            if latched || ground - feet > 1.0 * h0 {  // climber is well up the wall
                let line = ground - 0.6 * h0
                let thr = groundStrong ? line : weak32(line)
                for i in 0..<cc.count where cc.boxes[i].y1 > thr { bad[i] = true }  // people standing at the base
            }
        }
        return bad
    }

    // MARK: ground latch, tagged-on-the-wall check, tie-in phase

    /// Set while the lock is continuously the tagged/confirmed person and up; released only when the
    /// lock is followed back down.
    func latch(_ st: TrackState) {
        guard config.groundRule && config.groundLatch else { return }
        let box = predictedBox()
        let prevL = prev
        prev = (st, box)
        let up = ground - box.y1 > 1.0 * h0
        if st == .lost { cont = false }  // whoever is re-acquired later can't latch it; only force() can
        if st != .locked {
            jumpedSinceUp = jumpedSinceUp || st == .lost
            return
        }
        let followed = prevL != nil && prevL!.state == .locked && npIoU(prevL!.box, false, [box], false).v[0] > 0
        if up {
            latched = latched || cont
            upT = t
            jumpedSinceUp = false
        } else if !followed {
            jumpedSinceUp = true
        } else if latched && !jumpedSinceUp && t - upT <= 5.0 {
            latched = false
        }
    }

    func observe(_ t: Double, _ cc: Cands) {
        let live = people.indices.filter { t - people[$0].t <= 1.0 }
        var ids = [Int](repeating: -1, count: cc.count)
        if !live.isEmpty && cc.count > 0 {
            let lb = live.map { people[$0].box }
            var M = cc.boxes.map { npIoU($0, true, lb, false).v }
            while true {
                var bn = 0, bk = 0
                for n in M.indices { for k in lb.indices where M[n][k] > M[bn][bk] { bn = n; bk = k } }
                if !(M[bn][bk] >= 0.3) { break }
                people[live[bk]].t = t
                people[live[bk]].box = cc.boxes[bn]
                ids[bn] = live[bk]
                for k in lb.indices { M[bn][k] = -1 }
                for n in M.indices { M[n][bk] = -1 }
            }
        }
        for n in 0..<cc.count where ids[n] < 0 {
            let b = cc.boxes[n]
            people.append(Person(t0: t, t: t, box: b, box0: b, lock0: predictedBox()))
            ids[n] = people.count - 1
        }
        pid = ids
        if config.groundLatch && t - tTag <= 3.0 { taggedOnWall(cc) }
    }

    /// Someone beside the tag, at least 0.8x as tall, with their head below the tag's feet: the tag
    /// isn't standing on the floor. Their feet are the ground, and there is no tie-in to watch.
    func taggedOnWall(_ cc: Cands) {
        let tb = predictedBox()
        for b in cc.boxes {
            let gap = max(0.0, b.x0 - tb.x1, tb.x0 - b.x1) / max(tb.x1 - tb.x0, 1.0)
            if gap <= 1.0 && b.y0 >= tb.y1 && f32(b.y1 - b.y0) >= 0.8 * (tb.y1 - tb.y0) {
                let g = frameHeight != nil && b.y1 >= frameHeight! - 3 ? frameHeight! : b.y1
                if ground == groundTag || g < ground {
                    ground = g
                    groundStrong = false
                }
                tiein = false
            }
        }
    }

    func tieIn(_ t: Double, _ cc: Cands, _ s: Step) {
        let cfg = config
        let lb = predictedBox()
        let hl = max(lb.y1 - lb.y0, 1.0), lw = max(lb.x1 - lb.x0, 1.0)
        if t - tTag > cfg.tieinS || (cont && s.state == .locked && groundTag - lb.y1 > 2.0 * hl) {
            tiein = false  // timed out, or the lock itself left the ground
            people = []
            return
        }
        let app = cc.count > 0 ? tagProfile.sim(cc, cfg) : Scores(v: [], f32: false)
        for (n, i) in pid.enumerated() where app.v[n] >= 0 {
            people[i].apps = Array((people[i].apps + [app.v[n]]).suffix(10))
        }
        let lockI = s.chosen >= 0 && s.chosen < pid.count ? pid[s.chosen] : -1
        if lockI >= 0 && app.v[s.chosen] >= 0 { lockApps = Array((lockApps + [app.v[s.chosen]]).suffix(10)) }
        if groundTag - lb.y1 > 0.5 * hl { return }  // the lock isn't standing where the climber tied in
        var risers = [Int]()
        for i in pid {
            let p = people[i]
            if i == lockI || p.t0 < tTag { continue }
            let b = p.box, b0 = p.box0, L = p.lock0, h = p.box.y1 - p.box.y0
            let bornBeside = max(0.0, b0.x0 - L.x1, L.x0 - b0.x1) <= max(L.x1 - L.x0, 1.0)  // appeared beside the lock
            let up = b.y1 <= lb.y1 - cfg.riserFeetH * h && b.y0 <= lb.y0 - cfg.riserHeadH * h
                && h >= cfg.riserSize * hl && abs((b.x0 + b.x1) / 2 - (lb.x0 + lb.x1) / 2) <= cfg.riserColumn * lw
            if up && bornBeside {
                if people[i].upSince == nil { people[i].upSince = t }
                if t - people[i].upSince! >= cfg.riserHoldS { risers.append(i) }
            } else {
                people[i].upSince = nil
            }
        }
        let la = lockApps.isEmpty ? 0.0 : median(lockApps)
        let ok = risers.filter { !people[$0].apps.isEmpty && median(people[$0].apps) >= max(cfg.riserFloor, la + cfg.riserMargin) }
        guard ok.count == 1 else { return }  // nobody, or ambiguous: leave it to the referee
        let r = people[ok[0]]
        events.append(TrackerEvent(t: t, kind: "ascent swap", box: r.box))
        let lockFeet = lb.y1
        force(box: r.box, t: t)
        jumpT = t                // the referee, if any, checks the swap
        profile = tagProfile     // forget what was learned from the wrong person
        appLevel = 0.85
        if lockFeet > ground {   // where the lock stood is the ground...
            ground = lockFeet
            groundStrong = true
        }
        h0 = r.box.y1 - r.box.y0 // ...in the climber's units
        tiein = false
        people = []
    }

    // MARK: filter

    func update(_ cent: SIMD2<Double>, _ box: Box, _ dt: Double) {
        let pred = c + v * dt
        let r = cent - pred
        c = pred + 0.6 * r  // alpha-beta filter
        v = v + 0.15 * r / dt
        let k = Float(0.3 as Double)
        wh = 0.7 * wh + SIMD2(Double(k * Float(f32(box.x1 - box.x0))), Double(k * Float(f32(box.y1 - box.y0))))
    }

    func miss(_ pred: SIMD2<Double>, _ since: Double) -> Step {
        c = pred
        v *= 0.8
        state = since < config.uncertainS ? .uncertain : .lost
        return Step(state: state, box: predictedBox(), velocity: v, chosen: -1, app: 0, score: 0)
    }

    func reacquire(_ pred: SIMD2<Double>, _ cc: Cands, _ since: Double) -> Step {
        let (cents, hs) = cc.geometry()
        var app = profile.sim(cc, config)
        let bad = implausible(cc, hs, nil, false)
        let vet = vetoed(cc.count)
        for i in 0..<cc.count where bad[i] || vet[i] { app.v[i] = -1.0 }
        let best = argmax(app.v)
        if app.lt(best, config.reacquireApp) {
            reacq = nil
            return miss(pred, since)
        }
        if let p = reacq, norm32([Float(cents[best].x - p.cent.x), Float(cents[best].y - p.cent.y)]) < Float(hs[best]) {
            reacq = (cents[best], p.n + 1)
        } else {
            reacq = (cents[best], 1)
        }
        if reacq!.n < config.reacquireFrames { return miss(pred, since) }
        reacq = nil
        let b = cc.boxes[best]
        c = cents[best]
        v = .zero
        wh = SIMD2(f32(b.x1 - b.x0), f32(b.y1 - b.y0))
        state = .locked
        tSeen = t
        return Step(state: .locked, box: b, velocity: v, chosen: best, app: app.v[best], score: app.v[best], source: .spotter)
    }
}
