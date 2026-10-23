import Foundation

/// Tag window length (directe_tracking/online.py `run`): collect ticks from the tap until one is at least this many
/// seconds after the first, and include that one.
public let tagWindowSeconds = 1.5

extension TargetTracker {
    /// The tag moment as directe_tracking/online.py runs it (`_start` and the replay of the window): tag the
    /// climber in `window`, start a tracker at the tagged tick, and step every window tick (no camera
    /// calls). Returns the tracker, one step per window tick, and the tagged tick's index.
    public static func start(window: [(t: Double, candidates: [Candidate])], point: SIMD2<Double>, frameHeight: Double,
                             config: TrackerConfig = TrackerConfig()) throws -> (tracker: TargetTracker, steps: [Step], tagIndex: Int) {
        let (profile, box, k) = try tag(window: window, point: point, frameHeight: frameHeight)
        let trk = TargetTracker(profile: profile, box: box, t: window[k].t, config: config, frameHeight: frameHeight)
        let steps = window.map { trk.step(t: $0.t, view: $0.candidates) }
        return (trk, steps, k)
    }
}

public enum TagError: Error, Equatable {
    case emptyWindow
    /// Nobody under the tap (or within 0.1 frame heights of it) in any frame of the window.
    case noDetectionNearTap
}

/// The "tag climber" moment (lab tracker `tag`): pick the detection under the tap, follow it by
/// overlap through the window (longer than `seconds` if the body wasn't seen well enough, up to
/// `maxSeconds` or the end of the window), and average its appearance.
///
/// `window`: the ticks of the tag window in time order, each with its candidates (the app collects
/// 1.5 s from the tap time: every tick until and including the first one >= 1.5 s after the first,
/// as directe_tracking/online.py `run`). `point`: the tap in source pixels; `frameHeight`: source frame height.
/// `tSeed`: where the search starts (default: the first tick).
/// Returns the profile, the tagged box (float32-valued, from the first tick that had it) and the
/// index of that tick: start `TargetTracker(profile:box:t: window[index].t ...)`, then step every
/// window tick, then go on with live ticks.
public func tag(window: [(t: Double, candidates: [Candidate])], point: SIMD2<Double>, frameHeight: Double,
                seconds: Double = 1.5, maxSeconds: Double = 5.0, tSeed: Double? = nil) throws -> (Profile, Box, Int) {
    guard !window.isEmpty else { throw TagError.emptyWindow }
    let frames = window.map { Cands($0.candidates) }
    let ts = window.map(\.t)
    let seed = tSeed ?? ts[0]
    let i0 = argmin(ts.map { abs($0 - seed) })
    let px = point.x, py = point.y
    let px32 = weak32(px), py32 = weak32(py)  // compared with the float32 boxes in float32
    var found: (i: Int, k: Int)?
    for i in i0..<frames.count {
        let b = frames[i].boxes
        if b.isEmpty { continue }
        let inside = b.map { $0.x0 <= px32 && px32 <= $0.x1 && $0.y0 <= py32 && py32 <= $0.y1 }
        let cents = frames[i].geometry().cents
        let dist = b.indices.map { j -> Double in
            let dx = cents[j].x - px, dy = cents[j].y - py
            return (0 + dx * dx + dy * dy).squareRoot() - (inside[j] ? 1e6 : 0)
        }
        let k = argmin(dist)
        if inside[k] || dist[k] < 0.1 * frameHeight {
            found = (i, k)
            break
        }
    }
    guard let (start, first) = found else { throw TagError.noDetectionNearTap }
    var box = frames[start].boxes[first]
    var picked = [(start, first)]
    var j = start

    func evidence() -> Double {
        var seen = [Bool](repeating: false, count: Cands.R)
        for (i, k) in picked { for r in 0..<Cands.R where frames[i].valids[k * Cands.R + r] { seen[r] = true } }
        var s: Float = 0
        for r in 0..<Cands.R where seen[r] { s += AppearanceSpec.weights[r] }
        return Double(s)
    }

    while j + 1 < frames.count
            && (ts[j + 1] - ts[start] <= seconds || (evidence() < 0.85 && ts[j + 1] - ts[start] <= maxSeconds)) {
        j += 1
        let b = frames[j].boxes
        if b.isEmpty { continue }
        let ov = npIoU(box, true, b, true)
        let m = argmax(ov.v)
        if ov.v[m] > weak32(0.4) {
            box = b[m]
            picked.append((j, m))
        }
    }
    // Mean colour per region over the frames where it was seen (float32 sums, float64 division).
    var cnt = [Int](repeating: 0, count: Cands.R)
    var sum = [Float](repeating: 0, count: Cands.FD)
    for (i, k) in picked {
        let f = frames[i].feat(k), v = frames[i].valid(k)
        for r in 0..<Cands.R {
            if v[r] { cnt[r] += 1 }
            for d in r * Cands.D ..< (r + 1) * Cands.D { sum[d] = sum[d] + (v[r] ? f[d] : f[d] * 0) }
        }
    }
    let feat = (0..<Cands.FD).map { d in Float(Double(sum[d]) / Double(max(cnt[d / Cands.D], 1))) }
    var emb: [Float]?
    if picked.allSatisfy({ frames[$0.0].embs != nil && frames[$0.0].e > 0 }) {
        let e = frames[start].e
        var acc = [Float](repeating: 0, count: e)
        for (i, k) in picked {
            let row = frames[i].emb(k)
            precondition(row.count == e, "embedding sizes differ")
            for (d, x) in row.enumerated() { acc[d] = acc[d] + x }
        }
        let n = Float(picked.count)
        let mean = acc.map { $0 / n }
        let nrm = norm32(mean)
        let den = Float(1e-9 as Double) > nrm ? Float(1e-9 as Double) : nrm
        emb = mean.map { $0 / den }
    }
    return (Profile(feat: feat, valid: cnt.map { $0 > 0 }, emb: emb), frames[start].boxes[first], start)
}
