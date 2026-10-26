import Accelerate
import CoreVideo
import Foundation
import TrackerCore

/// Follow the climber through detector misses by continuity: port of directe_tracking/follow.py, mode "ncc"
/// (docs/review/deep-dives/far-climbers.md). They were confirmed here a moment ago, something that looks like that
/// patch is still here and moved lately, so it's still them.
///
/// Each look where the detector has nobody at the prediction: the patch of the last detector-confirmed climber (the
/// anchor, at a scale where the body is ~40 px) is matched (normalised cross-correlation, cv2 TM_CCOEFF_NORMED) in a
/// window of one body height round the last followed box. Accepted when the match is >= `nccMin` and the matched spot
/// moved within `aliveS` (frame difference, camera shift compensated; a static hold that correlates never moves), or
/// the match is >= `nccStill`. At most `maxCoastS` after the last confirmation, at most `maxStep` body heights a look,
/// never onto a box the detector found (that person is the tracker's call), nothing after a camera bump. The proposal
/// is one more candidate: the tracker still judges it (appearance, position), as with the second chance.
public final class MotionFollow {
    public var maxCoastS = 4.0, bodyPx = 40.0, nccMin: Float = 0.45, nccStill: Float = 0.7, aliveS = 1.5
    public var maxStep = 0.6, search = 1.0
    public private(set) var proposed = 0, accepted = 0

    /// A grey region of the frame (origin ox, oy); `at` takes frame coordinates, clamped into the region.
    private struct Grey {
        var w: Int, h: Int, px: [Float], ox = 0, oy = 0
        @inline(__always) func at(_ x: Int, _ y: Int) -> Float {
            px[min(max(y - oy, 0), h - 1) * w + min(max(x - ox, 0), w - 1)]
        }
    }
    private var anchor: (t: Double, box: Box, tpl: Grey, s: Double)?
    private var box: Box?
    private var tMoved = -1e9
    private var prev: (grey: Grey, offset: SIMD2<Double>)?
    private var proposal: Box?

    public init() {}

    /// Before the tracker's step, with this look's candidates. `offset`: the camera's accumulated image shift
    /// (Pipeline `offset`), so looks skipped in between are compensated too. Returns a box or nil.
    public func propose(t: Double, frame: CVPixelBuffer, offset: SIMD2<Double>, shiftKnown: Bool, predicted: Box,
                        boxes: [Box]) throws -> Box? {
        // grey only round where the climber can be: the followed box (or the prediction), three body heights out
        let around = box ?? predicted
        let m = 3 * max(around.height, 20)
        let g = try withPixels(frame) { v in
            Self.grey(v, Int(around.x0 - m), Int(around.y0 - m), Int(around.x1 + m), Int(around.y1 + m))
        }
        let last = prev
        prev = (g, offset)
        proposal = nil
        guard shiftKnown else { anchor = nil; return nil }  // bumped: the climber may be anywhere
        if let b = box, let last { box = b.offset(offset - last.offset) }
        guard let a = anchor, let b = box, let last, t - a.t <= maxCoastS else { return nil }
        if boxes.contains(where: { predicted.iou($0) >= 0.3 }) { return nil }  // the detector has someone there
        let h = b.height, w = b.width
        let r = search * h + 2
        let x0 = max(Int(b.x0 - r), g.ox), y0 = max(Int(b.y0 - r), g.oy)
        let x1 = min(Int(b.x1 + r), g.ox + g.w), y1 = min(Int(b.y1 + r), g.oy + g.h)
        guard Double(x1 - x0) >= w + 2, Double(y1 - y0) >= h + 2 else { return nil }
        // motion: this look minus the last, the last moved by the camera (whole pixels)
        let d = offset - last.offset
        let dx = Int(d.x.rounded()), dy = Int(d.y.rounded())
        let rw = x1 - x0, rh = y1 - y0
        var diff = [Float](repeating: 0, count: rw * rh)
        for y in 0..<rh {
            for x in 0..<rw {
                diff[y * rw + x] = abs(g.at(x0 + x, y0 + y) - last.grey.at(x0 + x - dx, y0 + y - dy))
            }
        }
        diff = gaussianBlur(diff, rw, rh, sigma: Float(max(0.6, 0.02 * h)))
        let med = medianOf(diff)
        let mad = medianOf(diff.map { abs($0 - med) })
        let thr = max(6, med + 5 * 1.4826 * mad)
        // match the anchor's patch at its scale
        let s = a.s
        let roi = Self.resize(g, x0, y0, rw, rh, s)
        guard roi.w > a.tpl.w, roi.h > a.tpl.h, let (v, mx, my) = Self.ncc(roi, a.tpl), v >= nccMin else { return nil }
        let bw = a.box.width, bh = a.box.height
        let nb = Box(Double(x0) + Double(mx) / s, Double(y0) + Double(my) / s,
                     Double(x0) + Double(mx) / s + bw, Double(y0) + Double(my) / s + bh)
        // alive: something moved inside the matched box's middle
        let mx0 = Int(nb.x0) - x0 + Int(0.15 * bw), my0 = Int(nb.y0) - y0 + Int(0.1 * bh)
        let mx1 = Int(nb.x1) - x0 - Int(0.15 * bw), my1 = Int(nb.y1) - y0 - Int(0.1 * bh)
        var moving = 0, n = 0
        for y in max(my0, 0)..<max(min(my1, rh), max(my0, 0)) {
            for x in max(mx0, 0)..<max(min(mx1, rw), max(mx0, 0)) {
                n += 1
                if diff[y * rw + x] > thr { moving += 1 }
            }
        }
        if n > 0, Double(moving) / Double(n) > 0.03 { tMoved = t }
        guard v >= nccStill || t - tMoved <= aliveS else { return nil }
        if (nb.center - b.center).norm > maxStep * h { return nil }
        if boxes.contains(where: { nb.iou($0) >= 0.3 }) { return nil }
        box = nb
        proposal = nb
        proposed += 1
        return nb
    }

    /// After the tracker's step: a lock on a detector box re-anchors; a lock on our proposal moves the follow.
    public func after(t: Double, frame: CVPixelBuffer, state: TrackState, chosen: Box?) throws {
        guard state == .locked, let c = chosen else {
            if state == .lost { box = nil }
            return
        }
        if let p = proposal, c.iou(p) > 0.9 { accepted += 1; return }  // coasting on our box: the anchor stays
        guard c.height >= 6 else { return }
        let g = try withPixels(frame) { v in Self.grey(v, Int(c.x0) - 1, Int(c.y0) - 1, Int(c.x1) + 2, Int(c.y1) + 2) }
        let s = min(max(bodyPx / c.height, 0.35), 3.0)
        let x0 = max(Int(c.x0.rounded()), g.ox), y0 = max(Int(c.y0.rounded()), g.oy)
        let x1 = min(Int(c.x1.rounded()), g.ox + g.w), y1 = min(Int(c.y1.rounded()), g.oy + g.h)
        guard x1 - x0 >= 3, y1 - y0 >= 3 else { return }
        anchor = (t, c, Self.resize(g, x0, y0, x1 - x0, y1 - y0, s), s)
        box = c
        tMoved = t
    }

    // MARK: pixels

    /// cv2.COLOR_BGR2GRAY (0.299 R + 0.587 G + 0.114 B) of a region (clipped to the frame), as floats.
    private static func grey(_ v: PixelView, _ ax: Int, _ ay: Int, _ bx: Int, _ by: Int) -> Grey {
        let x0 = max(ax, 0), y0 = max(ay, 0), x1 = max(min(bx, v.width), x0 + 1), y1 = max(min(by, v.height), y0 + 1)
        let w = x1 - x0, h = y1 - y0
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            let r = v.row(y0 + y)
            for x in 0..<w {
                let k = 4 * (x0 + x)
                out[y * w + x] = 0.114 * Float(r[k]) + 0.587 * Float(r[k + 1]) + 0.299 * Float(r[k + 2])
            }
        }
        return Grey(w: w, h: h, px: out, ox: x0, oy: y0)
    }

    /// A region scaled by s: area averaging when shrinking, bilinear when growing (cv2 INTER_AREA / INTER_LINEAR).
    private static func resize(_ g: Grey, _ x0: Int, _ y0: Int, _ w: Int, _ h: Int, _ s: Double) -> Grey {
        let ow = max(Int((Double(w) * s).rounded()), 1), oh = max(Int((Double(h) * s).rounded()), 1)
        var out = [Float](repeating: 0, count: ow * oh)
        for oy in 0..<oh {
            for ox in 0..<ow {
                if s < 1 {
                    let ax = x0 + Int(Double(ox) / s), bx = x0 + max(Int(Double(ox + 1) / s), Int(Double(ox) / s) + 1)
                    let ay = y0 + Int(Double(oy) / s), by = y0 + max(Int(Double(oy + 1) / s), Int(Double(oy) / s) + 1)
                    var sum: Float = 0, n: Float = 0
                    for y in ay..<min(by, g.oy + g.h) { for x in ax..<min(bx, g.ox + g.w) { sum += g.at(x, y); n += 1 } }
                    out[oy * ow + ox] = n > 0 ? sum / n : 0
                } else {
                    let fx = min(max((Double(ox) + 0.5) / s - 0.5, 0), Double(w - 1)) + Double(x0)
                    let fy = min(max((Double(oy) + 0.5) / s - 0.5, 0), Double(h - 1)) + Double(y0)
                    let ix = min(Int(fx), g.ox + g.w - 1), iy = min(Int(fy), g.oy + g.h - 1)
                    let jx = min(ix + 1, g.ox + g.w - 1), jy = min(iy + 1, g.oy + g.h - 1)
                    let ux = Float(fx - Double(ix)), uy = Float(fy - Double(iy))
                    let top = g.at(ix, iy) * (1 - ux) + g.at(jx, iy) * ux
                    let bot = g.at(ix, jy) * (1 - ux) + g.at(jx, jy) * ux
                    out[oy * ow + ox] = top * (1 - uy) + bot * uy
                }
            }
        }
        return Grey(w: ow, h: oh, px: out)
    }

    /// cv2.matchTemplate TM_CCOEFF_NORMED: the best position and its score; nil for a flat template.
    private static func ncc(_ img: Grey, _ tpl: Grey) -> (Float, Int, Int)? {
        let tw = tpl.w, th = tpl.h, n = Float(tw * th)
        let mean = tpl.px.reduce(0, +) / n
        let t0 = tpl.px.map { $0 - mean }
        var tss: Float = 0
        vDSP_svesq(t0, 1, &tss, vDSP_Length(t0.count))
        guard tss.squareRoot() / n.squareRoot() >= 3 else { return nil }
        var best: (Float, Int, Int) = (-2, 0, 0)
        var row = [Float](repeating: 0, count: tw)
        for y in 0...(img.h - th) {
            for x in 0...(img.w - tw) {
                var cross: Float = 0, sum: Float = 0, sq: Float = 0
                for ty in 0..<th {
                    let base = (y + ty) * img.w + x
                    img.px.withUnsafeBufferPointer { p in
                        for k in 0..<tw { row[k] = p[base + k] }
                    }
                    var c: Float = 0, s: Float = 0, q: Float = 0
                    t0.withUnsafeBufferPointer { tp in vDSP_dotpr(tp.baseAddress! + ty * tw, 1, row, 1, &c, vDSP_Length(tw)) }
                    vDSP_sve(row, 1, &s, vDSP_Length(tw))
                    vDSP_svesq(row, 1, &q, vDSP_Length(tw))
                    cross += c; sum += s; sq += q
                }
                let varI = sq - sum * sum / n
                guard varI > 1e-3 else { continue }
                let v = cross / (tss * varI).squareRoot()
                if v > best.0 { best = (v, x, y) }
            }
        }
        return best.0 > -2 ? best : nil
    }
}

private extension SIMD2 where Scalar == Double {
    var norm: Double { (x * x + y * y).squareRoot() }
}
