import Foundation

/// Everyone else's short position track (TrackerConfig.bystanders). On a confident frame everyone but the climber is
/// "not her"; following each of them by position keeps that label on them as they move, so someone who is just one of
/// those people carrying on can't take the lock, whatever they look like. Position never overrules appearance where
/// two tracks could have swapped: a real overlap with the climber or with someone unproven, the climber unseen next to
/// them, or a track picked up again after a gap loses the label; someone only brushing the climber has it set aside
/// until they're apart again.
struct Bystanders {
    struct Track {
        var box: Box
        /// px/s.
        var v = SIMD2<Double>(0, 0)
        var seen: Double
        /// Seconds earned as "not her" on confident frames, apart from the climber.
        var notHer = 0.0
        /// Beside the climber at the last look: the label is set aside.
        var touching = false
    }

    var tracks: [Track] = []
    /// Per candidate of the last step: its track (-1: none).
    var ids: [Int] = []
    var t: Double?

    mutating func camera(_ shift: SIMD2<Double>) {
        for i in tracks.indices { tracks[i].box = tracks[i].box.offset(shift) }
    }

    /// Takes this frame's people onto the tracks one to one, best overlap with where each track was heading first; a
    /// second box of someone matched shares their track, anyone else starts one. Tracks unseen for `keepS` end.
    mutating func associate(_ t: Double, _ boxes: [Box], _ cfg: TrackerConfig) {
        let dt = self.t.map { max(t - $0, 0) } ?? 0
        self.t = t
        let pred = tracks.map { tr -> Box in
            let gone = t - tr.seen
            return tr.box.offset(tr.v * min(gone, cfg.bystanderKeepS))
        }
        let n = boxes.count
        var got = [Int](repeating: -1, count: n)
        var taken = [Bool](repeating: false, count: tracks.count)
        if !tracks.isEmpty && n > 0 {
            var M = boxes.map { b in pred.map { b.iou($0) } }
            while true {
                var bi = 0, bj = 0, best = -Double.infinity
                for i in 0..<n { for j in tracks.indices where M[i][j] > best { best = M[i][j]; bi = i; bj = j } }
                if !(best > cfg.bystanderIoU) { break }
                got[bi] = bj
                taken[bj] = true
                for j in tracks.indices { M[bi][j] = -1 }
                for i in 0..<n { M[i][bj] = -1 }
            }
        }
        for i in 0..<n where got[i] < 0 {  // the pose and the mask box of one person
            if let k = (0..<n).first(where: { got[$0] >= 0 && boxes[$0].samePerson(boxes[i]) }) { got[i] = got[k] }
        }
        var next: [Track] = []
        var map = [Int](repeating: -1, count: tracks.count)
        for j in tracks.indices {
            if taken[j] {
                let i = (0..<n).first { got[$0] == j }!
                var tr = tracks[j]
                let gone = t - tr.seen
                if gone > 0 { tr.v = 0.7 * tr.v + 0.3 * (boxes[i].center - tr.box.center) / gone }
                if gone > cfg.bystanderGapS { tr.notHer = 0 }  // picked up again after a gap: a guess
                tr.box = boxes[i]
                tr.seen = t
                map[j] = next.count
                next.append(tr)
            } else if t - tracks[j].seen <= cfg.bystanderKeepS {
                var tr = tracks[j]
                tr.v *= pow(0.8, dt * 10)
                map[j] = next.count
                next.append(tr)
            }
        }
        ids = got.map { $0 >= 0 ? map[$0] : -1 }
        for i in 0..<n where ids[i] < 0 {
            if let k = (0..<i).first(where: { ids[$0] >= 0 && got[$0] < 0 && boxes[$0].samePerson(boxes[i]) }) {
                ids[i] = ids[k]
                continue
            }
            ids[i] = next.count
            next.append(Track(box: boxes[i], seen: t))
        }
        tracks = next
    }

    /// Per candidate: someone who earned "not her" for `bystanderS` and wasn't beside the climber at the last look.
    func ruled(_ cfg: TrackerConfig) -> [Bool] {
        ids.map { $0 >= 0 && tracks[$0].notHer >= cfg.bystanderS && !tracks[$0].touching }
    }

    /// After the step: on a confident frame (`chosen` >= 0) everyone else seen apart from the climber earns `dt`; the
    /// climber's own track (and a second box of them) never does. Near the climber's place (`near`: the lock's box when
    /// `observed`, else where they were predicted) a label is only kept while the lock is seen and the two boxes barely
    /// touch (overlap under `bystanderCross`): it's set aside until they're apart again. A real overlap, the climber
    /// unseen next to them, or vanishing next to the climber (they may have merged into one box) loses it, and so does a
    /// labelled track crossing someone who hasn't earned it.
    mutating func after(near: Box, observed: Bool, chosen: Int, confident: Bool, dt: Double, _ cfg: TrackerConfig) {
        let h = max(near.height, 1)
        let zone = Box(near.x0 - cfg.bystanderNear * h, near.y0 - cfg.bystanderNear * h,
                       near.x1 + cfg.bystanderNear * h, near.y1 + cfg.bystanderNear * h)
        let now = t ?? 0
        let own = chosen >= 0 && chosen < ids.count ? ids[chosen] : -1
        let at = tracks.map { $0.box.offset($0.v * (now - $0.seen)) }  // where each is now, or would be
        var reset = Set<Int>()
        for j in tracks.indices {
            let live = tracks[j].seen == now
            tracks[j].touching = zone.iou(at[j]) > 0
            if tracks[j].touching && (!live || !observed || j == own || near.iou(at[j]) >= cfg.bystanderCross) { reset.insert(j) }
        }
        for a in tracks.indices where tracks[a].notHer >= cfg.bystanderS {
            for b in tracks.indices where b != a && tracks[b].seen == now && tracks[b].notHer < cfg.bystanderS
                && at[a].iou(at[b]) >= cfg.bystanderCross {
                reset.insert(a)  // someone unproven crossed them
            }
        }
        if own >= 0 { reset.insert(own) }
        for j in tracks.indices {
            if reset.contains(j) { tracks[j].notHer = 0; continue }
            if confident, tracks[j].seen == now, !tracks[j].touching { tracks[j].notHer += min(dt, 0.5) }
        }
    }

    /// The lock was wrong (referee release/force, ascent swap): what it called "not her" may be her.
    mutating func forget() {
        for j in tracks.indices { tracks[j].notHer = 0 }
    }
}
