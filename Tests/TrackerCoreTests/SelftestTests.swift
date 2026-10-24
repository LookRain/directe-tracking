import Foundation
import Testing
@testable import TrackerCore

// directe_tracking/tracker.py `selftest` (python -m lab.tracker), scene for scene: the jump guard on synthetic
// scenes where everyone looks identical, so only geometry decides; a look-alike stacked below the climber
// (Config.switch_margin); then the tie-in phase.

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

/// A look-alike climbs a body length below on the next route, seen steeply from the side: the boxes overlap. With
/// `gaps` the detector misses the look-alike for a frame, then the climber for 0.4 s; without, the climber's colour
/// reads wrong for a frame (torso and legs). True if the lock ever moved onto the look-alike.
private func stackedTookLock(_ cfg: TrackerConfig, gaps: Bool) -> Bool {
    let trk = tracker(cfg)
    var s: Step!, y = 600.0
    for k in 1..<50 {
        let t = Double(k) * 0.1
        y = 600 - 0.25 * H * t
        var pts: [(Double, Double)] = gaps && 2.0 <= t && t < 2.4 ? [] : [(500, y)]
        if !(gaps && abs(t - 1.9) < 1e-6) { pts.append((540, y + 0.8 * H)) }
        var c = cands(pts)
        if !gaps && abs(t - 3.0) < 1e-6 {
            var f = Array(repeating: Float(0), count: D)
            f[3] = 1
            for r in 1..<R { c[0].feats[r] = f }
        }
        s = trk.step(t: t, view: c)
        if on(s, 540, y + 0.8 * H) { return true }
    }
    #expect(on(s, 500, y), "didn't keep the climber: \(s.state) \(s.box)")
    return false
}

@Test func selftestStacked() {
    var off = TrackerConfig()
    off.switchMargin = -1
    off.trackKeepS = 0
    #expect(stackedTookLock(off, gaps: true) && stackedTookLock(off, gaps: false), "test is too easy: the look-alike never took the lock")
    var noKeep = TrackerConfig()
    noKeep.trackKeepS = 0
    #expect(stackedTookLock(noKeep, gaps: true), "test is too easy: without remembering missed people it held anyway")
    #expect(!stackedTookLock(TrackerConfig(), gaps: true), "a look-alike stacked below took the lock while the climber was missed")
    #expect(!stackedTookLock(TrackerConfig(), gaps: false), "a look-alike stacked below took the lock when the climber's colour wobbled")
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

// Reusing an earlier appearance (as the Python selftest's last scene): it adds what agrees with the fresh tag, leaves
// out views and negatives that disagree with it, and nothing at all when the anchors disagree.
@Test func reusingAnEarlierAppearance() throws {
    var seed: UInt64 = 3
    func random() -> Float {  // a small LCG: the scene, not the Python one's numbers
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return Float(Double(seed >> 11) / Double(1 << 53)) * 2 - 1
    }
    func unit(_ v: [Float]) -> [Float] { let n = v.map { $0 * $0 }.reduce(0, +).squareRoot(); return v.map { $0 / n } }
    func noise() -> [Float] { unit((0..<64).map { _ in random() }) }
    let base = noise()
    func near(_ k: Float) -> [Float] { unit(zip(base, noise()).map { $0 + k * $1 }) }
    func cos(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).map(*).reduce(0, +) }
    let c = cands([(500, 600)])[0]
    let fresh = Profile(feats: c.feats, valid: c.valids, emb: near(0.2))
    var prior = Profile(feats: c.feats, valid: c.valids, emb: near(0.2))
    prior.gallery = [near(0.8), near(0.8), noise()]  // the last: someone else, a polluted view
    prior.negs = [noise(), near(0.2)]                // the last: the climber, banked by mistake
    prior.negF = [c.feats.flatMap { $0 }, c.feats.flatMap { $0 }]
    prior.negV = [c.valids, c.valids]
    let cfg = TrackerConfig()
    let r = fresh.reusing(prior, config: cfg)
    #expect(r.gallery.count == 3 && cos(r.gallery.last!, prior.emb!) > 0.99, "the agreeing views, then the old anchor")
    #expect(r.gallery.allSatisfy { Double(cos($0, fresh.emb!)) >= cfg.reidLo + cfg.galleryAdmit })
    #expect(r.negs.count == 1, "a banked negative that passes for the tapped climber stays out")
    #expect(r.emb == fresh.emb && r.feat == fresh.feat, "the tap stays the anchor")
    let stranger = Profile(feats: c.feats, valid: c.valids, emb: noise())
    #expect(fresh.reusing(stranger, config: cfg).gallery.isEmpty, "an earlier appearance that disagrees adds nothing")
    let back = try JSONDecoder().decode(AppearanceState.self, from: JSONEncoder().encode(AppearanceState(r))).profile
    #expect(back.gallery == r.gallery && back.negs == r.negs && back.adapt == r.adapt)
}
