import Foundation
import Testing
@testable import TrackerCore

// TrackerConfig.bystanders on synthetic scenes where everyone looks identical, so only position decides.

private let H = 200.0  // body height, px
private let R = AppearanceSpec.regions.count, D = AppearanceSpec.dim

private func cands(_ centres: [(Double, Double)]) -> [Candidate] {
    centres.map { x, y in
        Candidate(box: Box(x - 0.3 * H, y - H / 2, x + 0.3 * H, y + H / 2),
                  feats: Array(repeating: Array(repeating: Float(1.0 / Double(D)), count: D), count: R),
                  valids: Array(repeating: true, count: R), emb: nil)
    }
}

private func tracker(bystanders: Bool) -> TargetTracker {
    var cfg = TrackerConfig()
    cfg.groundRule = false
    cfg.keepApart = false  // the older "seen beside the climber" rule would rule them out too
    cfg.bystanders = bystanders
    let c = cands([(500, 600)])[0]
    return TargetTracker(profile: Profile(feats: c.feats, valid: c.valids), box: c.box, t: 0, config: cfg)
}

/// Locked for 2 s with someone two body heights away, then the climber is gone for 4 s while that person climbs on:
/// whether the lock ever moves onto them.
private func tookSomeoneElse(bystanders: Bool) -> Bool {
    let trk = tracker(bystanders: bystanders)
    for k in 1...60 {
        let t = Double(k) * 0.1
        let other = (500 + 2 * H, 600 - 20 * t)
        let s = trk.step(t: t, view: cands((t < 2 ? [(500, 600)] : []) + [other]))
        if s.state == .locked && abs(s.box.center.x - other.0) < 0.3 * H { return true }
    }
    return false
}

@Test func bystanderCarryingOnIsRuledOut() {
    #expect(tookSomeoneElse(bystanders: false), "test is too easy: without bystander tracks nobody took the lock")
    #expect(!tookSomeoneElse(bystanders: true), "someone tracked apart from the climber took the lock once they were lost")
}

@Test func touchingSetsTheLabelAsideAndCrossingLosesIt() {
    let trk = tracker(bystanders: true)
    var t = 0.0, x = 500 + 2 * H
    func walk(to target: Double) {  // 0.1 body heights a tick, as someone climbing sideways
        repeat {
            x += max(-0.1 * H, min(0.1 * H, target - x))
            t += 0.1
            _ = trk.step(t: t, view: cands([(500, 600), (x, 600)]))
        } while abs(x - target) > 1e-6
    }
    walk(to: 500 + 2 * H)
    for _ in 0..<20 { walk(to: 500 + 2 * H + 0.01) ; walk(to: 500 + 2 * H) }  // apart for 4 s: earned
    #expect(trk.bystanderRuled == [false, true])
    walk(to: 500 + 0.55 * H)  // brushing the climber (boxes just overlap): set aside, not lost
    walk(to: 500 + 0.55 * H + 0.01)  // (a look's rule-out uses where everyone was at the last look)
    #expect(trk.bystanderRuled == [false, false])
    walk(to: 500 + 2 * H)  // apart again: back
    #expect(trk.bystanderRuled == [false, true])
    walk(to: 500 + 0.1 * H)  // a real overlap: they may have swapped, so it's gone...
    walk(to: 500 + 0.8 * H)
    #expect(trk.bystanderRuled == [false, false])
    walk(to: 500 + 2 * H)  // ...until earned again apart
    #expect(trk.bystanderRuled == [false, true])
}
