import Testing
@testable import TrackerCore

// directe_tracking/tracker.py `selftest` (python -m lab.tracker), scene for scene: the jump guard on synthetic
// scenes where everyone looks identical, so only geometry decides; then the tie-in phase.

private let H = 200.0  // body height, px
private let R = AppearanceSpec.regions.count, D = AppearanceSpec.dim

private func cands(_ centres: [(Double, Double)]) -> [Candidate] {
    centres.map { x, y in
        Candidate(box: Box(x - 0.3 * H, y - H / 2, x + 0.3 * H, y + H / 2),
                  feats: Array(repeating: Array(repeating: Float(1.0 / Double(D)), count: D), count: R),
                  valids: Array(repeating: true, count: R), emb: nil)
    }
}

private func tracker(_ cfg: TrackerConfig = TrackerConfig()) -> TargetTracker {
    let c = cands([(500, 600)])[0]
    return TargetTracker(profile: Profile(feats: c.feats, valid: c.valids), box: c.box, t: 0, config: cfg)
}

private func on(_ s: Step, _ x: Double, _ y: Double) -> Bool {
    s.state == .locked && abs(s.box.center.x - x) < 0.3 * H && abs(s.box.center.y - y) < 0.3 * H
}

@Test(arguments: [10.0, 5.0])
func selftestLongFallIsFollowed(fps: Double) {
    // 1.2 s free fall (~7 m) from t = 1 s, then caught; a bystander stands below
    let g = 9.8 / 1.7 * H  // px/s^2 for a 1.7 m climber
    var cfg = TrackerConfig()
    cfg.groundRule = false
    let trk = tracker(cfg), dt = 1.0 / fps
    for k in 1..<Int(3 * fps) {
        let t = Double(k) * dt
        let tf = min(max(t - 1.0, 0.0), 1.2)
        let y = 600 + 0.5 * g * tf * tf
        let s = trk.step(t: t, view: cands([(500, y), (500 - 3 * H, 600 + 3 * H)]))
        #expect(on(s, 500, y), "lost the falling climber at \(fps) fps, t=\(t): \(s.state) \(s.box)")
    }
}

/// Climber hidden for 1.5 s; an identical person climbs 1.3 body heights to the side.
private func lookalikeTookLock(_ cfg: TrackerConfig) -> Bool {
    let trk = tracker(cfg)
    var s: Step!
    for k in 1..<40 {
        let t = Double(k) * 0.1
        let hidden = 1.0 <= t && t < 2.5
        s = trk.step(t: t, view: cands((hidden ? [] : [(500, 600)]) + [(500 - 1.3 * H, 600)]))
        if on(s, 500 - 1.3 * H, 600) { return true }
    }
    #expect(on(s, 500, 600), "didn't get the climber back: \(s.state) \(s.box)")
    return false
}

@Test func selftestLookalike() {
    var unguarded = TrackerConfig()
    unguarded.jumpGuard = false
    #expect(lookalikeTookLock(unguarded), "test is too easy: the unguarded tracker didn't switch")
    #expect(!lookalikeTookLock(TrackerConfig()), "a lookalike took the lock while the climber was hidden")
}

@Test(arguments: [true, false])
func selftestCameraBump(known: Bool) {
    // someone knocks the phone at k = 15; the shift is either measured or unknown (conf 0)
    let trk = tracker()
    var off = SIMD2<Double>(0, 0)
    for k in 1..<40 {
        let t = Double(k) * 0.1
        let shift = k == 15 ? SIMD2<Double>(300, -250) : .zero
        off += shift
        trk.camera(shift: known ? shift : .zero, conf: known || k != 15 ? 0.9 : 0.0)
        let s = trk.step(t: t, view: cands([(500 + off.x, 600 + off.y), (500 - 1.3 * H + off.x, 600 + off.y)]))
        #expect(!on(s, 500 - 1.3 * H + off.x, 600 + off.y), "bump (known=\(known)) gave the lock away at t=\(t)")
        if known {
            #expect(on(s, 500 + off.x, 600 + off.y), "known bump lost the climber at t=\(t): \(s.state)")
        }
    }
}

/// Tie-in: climber A (colour 0) beside belayer B (mostly colour 5). If the lock slid onto B while A
/// was hidden, A climbing up beside B must take it back. If the lock stayed on A (a queue: B starts
/// up the same line), it must not move.
private func tieinEndsOnClimber(slid: Bool, tieinS: Double = 60) -> (Bool, [TrackerEvent]) {
    func people(_ pts: [(Double, Double)], _ cols: [[(Int, Float)]]) -> [Candidate] {
        var c = cands(pts)
        for (i, col) in cols.enumerated() {
            var f = Array(repeating: Float(0), count: D)
            for (k, w) in col { f[k] = w }
            c[i].feats = Array(repeating: f, count: R)
        }
        return c
    }
    let A: [(Int, Float)] = [(0, 1.0)], B: [(Int, Float)] = [(0, 0.6), (5, 0.4)]  // B: a lookalike to the small model (~0.77)
    let tagged = people([(500, 600)], [A])[0]
    var cfg = TrackerConfig()
    cfg.tieinS = tieinS
    let trk = TargetTracker(profile: Profile(feats: tagged.feats, valid: tagged.valids), box: tagged.box, t: 0, config: cfg)
    var a = (500.0, 600.0), b = (660.0, 600.0)
    var s: Step!
    for k in 1..<80 {
        let t = Double(k) * 0.1
        if slid && abs(t - 1.0) < 1e-6 {
            trk.force(box: people([b], [B])[0].box, t: t)  // the lock slides onto the belayer
        }
        if t >= 2.0 {
            if slid { a = (560.0, 600.0 - 1.0 * H * (t - 2.0)) } else { b = (660.0, 600.0 - 1.0 * H * (t - 2.0)) }  // queue: B climbs
        }
        let hidden = slid && 1.0 <= t && t < 2.0
        s = trk.step(t: t, view: people((hidden ? [] : [a]) + [b], (hidden ? [] : [A]) + [B]))
    }
    return (on(s, a.0, a.1), trk.events)
}

@Test func selftestTieIn() {
    #expect(!tieinEndsOnClimber(slid: true, tieinS: 0).0, "test is too easy: without the tie-in phase the lock came back anyway")
    let (back, events) = tieinEndsOnClimber(slid: true)
    #expect(back, "the climber left the ground beside the belayer, but the lock stayed")
    #expect(events.map(\.kind) == ["ascent swap"])
    #expect(tieinEndsOnClimber(slid: false).0, "queue: someone else started up beside the climber and took the lock")
}
