import Accelerate
import CoreVideo
import Foundation
import Vision

/// Is the wall moving here? Port of directe_tracking/motion.py Evidence, for `Perceiver.secondChance(motion:)`.
///
/// Fixed camera: a causal background model on a 270x480 copy (cv2 INTER_AREA), log-ratio
/// illumination-normalised (log(x + 8) minus its Gaussian blur, sigma 25), sigma-delta approximate
/// median (8x faster for the first 50 updates), not learned under the predicted climber. Moving camera:
/// difference with the frame ~1 s earlier, registered onto this one; the lab registers with ORB +
/// RANSAC, here Vision's homographic registration plays that role. Motion blobs are evidence only,
/// never candidates. Updates at most `hz` times a second.
///
/// The runner resets it when a fixed phone is knocked (lab: shift > 6 W/1080 px or camera-motion
/// response < 0.1): `update(..., reset: true)`.
///
/// Lazy: an update only keeps the small copy (and, for a fixed camera, learns the background); the
/// difference image and its blobs are worked out when `blobs` / `supports` is first asked after it
/// (only the second chance asks, and rarely), from exactly what the update saw. Same blobs as working
/// them out on every update, a fraction of the CPU (the sigma-25 blur, Vision registration and the
/// median were most of a tick's CPU time).
public final class WallMotion: MotionEvidence {
    public static let width = 270, height = 480
    public let movingCamera: Bool
    public let interval: Double
    /// Work the blobs out when asked (true) or on every update like the lab (false; same blobs, for comparison).
    public let lazy: Bool

    /// Motion blobs of the last update, in source pixels.
    public var blobs: [Box] {
        if let p = pending {
            pending = nil
            found = p()
        }
        return found
    }

    private var found: [Box] = []
    private var pending: (() -> [Box])?
    private var t = -1e9
    private var bg: [Float]?
    private var nUpdates = 0
    private var history: [Sample] = []
    private let sigma: Float = 25, delta: Float = 0.004

    /// One moving-camera update's picture: the small copy, its grey (for registration) and, once a difference
    /// needs it, the illumination-normalised image.
    private final class Sample {
        let t: Double
        let small: BGRAImage
        let grey: CVPixelBuffer
        let sigma: Float
        private var _n: [Float]?

        init(t: Double, small: BGRAImage, grey: CVPixelBuffer, sigma: Float) {
            self.t = t; self.small = small; self.grey = grey; self.sigma = sigma
        }

        var n: [Float] {
            if let v = _n { return v }
            let v = WallMotion.normalise(small.view, sigma: sigma)
            _n = v
            return v
        }
    }

    public init(movingCamera: Bool, hz: Double = 5, lazy: Bool = true) {
        self.movingCamera = movingCamera
        interval = 1 / hz
        self.lazy = lazy
    }

    public func supports(_ box: Box) -> Bool {
        for b in blobs {
            let c = b.center
            if box.iou(b) >= 0.1 || (box.x0 <= c.x && c.x <= box.x1 && box.y0 <= c.y && c.y <= box.y1) { return true }
        }
        return false
    }

    /// lab Evidence.update(t, frame, pred_box, reset).
    public func update(t now: Double, frame: CVPixelBuffer, predicted pb: Box, reset: Bool = false) throws {
        if reset {
            bg = nil; nUpdates = 0
            history.removeAll()
            found = []; pending = nil
        }
        if now - t < interval { return }
        t = now
        defer { if !lazy { _ = blobs } }
        let SW = Self.width, SH = Self.height
        let small = BGRAImage(width: SW, height: SH)
        let (fw, fh) = try withPixels(frame) { v -> (Int, Int) in
            cvResize(v, small.view, .area)
            return (v.width, v.height)
        }
        let sx = Double(SW) / Double(fw), sy = Double(SH) / Double(fh)
        let h = max(pb.height * sy, 8)
        found = []; pending = nil
        if movingCamera {
            guard let (cur, old) = remember(now, small) else { return }
            pending = { Self.blobBoxes(Self.registeredDiff(cur, old), hExp: h, sx: sx, sy: sy) }
            return
        }
        let n = Self.normalise(small.view, sigma: sigma)
        var d: [Float]?
        if let bg, nUpdates >= 50 {
            var out = [Float](repeating: 0, count: SW * SH)
            for i in 0..<(SW * SH) {
                out[i] = max(abs(n[3 * i] - bg[3 * i]), abs(n[3 * i + 1] - bg[3 * i + 1]), abs(n[3 * i + 2] - bg[3 * i + 2]))
            }
            d = out
        }
        // learn, except under the predicted climber
        let p = 0.5 * h
        let fx0 = max(Int(pb.x0 * sx - p), 0), fx1 = max(Int(pb.x1 * sx + p), 0)
        let fy0 = max(Int(pb.y0 * sy - p), 0), fy1 = max(Int(pb.y1 * sy + p), 0)
        if bg == nil {
            bg = n; nUpdates = 1
        } else {
            let step = delta * (nUpdates < 50 ? 8 : 1)
            for y in 0..<SH {
                let frozenRow = y >= fy0 && y < fy1
                for x in 0..<SW {
                    if frozenRow && x >= fx0 && x < fx1 { continue }
                    for c in 0..<3 {
                        let k = 3 * (y * SW + x)
                        let diff = n[k + c] - bg![k + c]
                        bg![k + c] += diff > 0 ? step : (diff < 0 ? -step : 0)
                    }
                }
            }
            nUpdates += 1
        }
        guard let d else { return }
        pending = { Self.blobBoxes(d, hExp: h, sx: sx, sy: sy) }
    }

    /// Blobs of a difference image (working px), in source pixels.
    private static func blobBoxes(_ d: [Float]?, hExp h: Double, sx: Double, sy: Double) -> [Box] {
        guard let d else { return [] }
        let blurred = gaussianBlur(d, width, height, sigma: 1)
        return blobs(blurred, width, height, hExp: h).map { Box($0.x0 / sx, $0.y0 / sy, $0.x1 / sx, $0.y1 / sy) }
    }

    /// log(x + 8) minus its Gaussian blur (sigma 25), per channel; interleaved B G R floats.
    static func normalise(_ v: PixelView, sigma: Float) -> [Float] {
        let w = v.width, h = v.height
        var out = [Float](repeating: 0, count: w * h * 3)
        for c in 0..<3 {
            var plane = [Float](repeating: 0, count: w * h)
            for y in 0..<h { let r = v.row(y); for x in 0..<w { plane[y * w + x] = log(Float(r[4 * x + c]) + 8) } }
            let blur = gaussianBlur(plane, w, h, sigma: sigma)
            for i in 0..<(w * h) { out[3 * i + c] = plane[i] - blur[i] }
        }
        return out
    }

    // MARK: moving camera

    /// Keeps this update's picture in the ~1 s history; returns it and the one ~1 s earlier to difference with, or
    /// nil when there's none old enough yet.
    private func remember(_ now: Double, _ small: BGRAImage) -> (Sample, Sample)? {
        let SW = Self.width, SH = Self.height
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, SW, SH, kCVPixelFormatType_OneComponent8, nil, &pb)
        guard let grey = pb else { return nil }
        CVPixelBufferLockBaseAddress(grey, [])
        let g = CVPixelBufferGetBaseAddress(grey)!.assumingMemoryBound(to: UInt8.self), rb = CVPixelBufferGetBytesPerRow(grey)
        let sv = small.view
        for y in 0..<SH { let r = sv.row(y); for x in 0..<SW { g[y * rb + x] = cvGrey(r[4 * x], r[4 * x + 1], r[4 * x + 2]) } }
        CVPixelBufferUnlockBaseAddress(grey, [])
        let cur = Sample(t: now, small: small, grey: grey, sigma: sigma)
        history.append(cur)
        while history.count > 2 && history[1].t <= now - 1.0 { history.removeFirst() }
        let old = history[0]
        if now - old.t < 0.9 { return nil }
        return (cur, old)
    }

    /// The registered difference of `cur` with `old` (moving camera), zero outside the area both see.
    private static func registeredDiff(_ cur: Sample, _ old: Sample) -> [Float]? {
        let SW = width, SH = height
        let grey = cur.grey, n = cur.n
        // Vision: warpTransform maps the targeted (new) image onto the reference (old) one, in pixels
        // with the origin at the bottom left.
        let req = VNHomographicImageRegistrationRequest(targetedCVPixelBuffer: grey, options: [:])
        do { try VNImageRequestHandler(cvPixelBuffer: old.grey, options: [:]).perform([req]) } catch { return nil }
        guard let obs = req.results?.first as? VNImageHomographicAlignmentObservation else { return nil }
        let m = obs.warpTransform
        let Hf = Float(SH)
        func toOld(_ x: Float, _ y: Float) -> (Float, Float)? {
            let yb = Hf - y
            let X = m[0][0] * x + m[1][0] * yb + m[2][0], Y = m[0][1] * x + m[1][1] * yb + m[2][1]
            let Z = m[0][2] * x + m[1][2] * yb + m[2][2]
            if abs(Z) < 1e-9 { return nil }
            return (X / Z, Hf - Y / Z)
        }
        let oldN = old.n
        var d = [Float](repeating: 0, count: SW * SH)
        var valid = [UInt8](repeating: 0, count: SW * SH)
        for y in 0..<SH {
            for x in 0..<SW {
                guard let (ox, oy) = toOld(Float(x) + 0.5, Float(y) + 0.5) else { continue }
                let fx = ox - 0.5, fy = oy - 0.5
                if fx < 0 || fy < 0 || fx > Float(SW - 1) || fy > Float(SH - 1) { continue }
                valid[y * SW + x] = 1
                let x0 = min(Int(fx), SW - 2), y0 = min(Int(fy), SH - 2)
                let ax = fx - Float(x0), ay = fy - Float(y0)
                var best: Float = 0
                for c in 0..<3 {
                    func at(_ xx: Int, _ yy: Int) -> Float { oldN[3 * (yy * SW + xx) + c] }
                    let w = (1 - ay) * ((1 - ax) * at(x0, y0) + ax * at(x0 + 1, y0)) + ay * ((1 - ax) * at(x0, y0 + 1) + ax * at(x0 + 1, y0 + 1))
                    best = max(best, abs(n[3 * (y * SW + x) + c] - w))
                }
                d[y * SW + x] = best
            }
        }
        // erode the valid area by 7x7, zero the difference outside it
        let er = morph(valid, SW, SH, radius: 3, dilate: false)
        for i in 0..<(SW * SH) where er[i] == 0 { d[i] = 0 }
        return d
    }

    // MARK: blobs

    /// lab blobs(d, h_exp): foreground at body-height scale (working px).
    static func blobs(_ d: [Float], _ w: Int, _ h: Int, hExp: Double, k: Float = 6, thrAbs: Float = 0.25) -> [Box] {
        let med = medianOf(d)
        let mad = medianOf(d.map { abs($0 - med) }) * 1.4826
        let thr = max(thrAbs, med + k * mad)
        var fg = d.map { $0 > thr ? UInt8(1) : 0 }
        fg = morph(morph(fg, w, h, radius: 1, dilate: true), w, h, radius: 1, dilate: false)  // MORPH_CLOSE 3x3
        let minArea = max(4, Int(0.004 * hExp * hExp))
        var boxes = components(fg, w, h).filter { $0.area >= minArea }.map { [Double($0.x0), Double($0.y0), Double($0.x1 + 1), Double($0.y1 + 1)] }
        // merge components within 0.35 body heights (a climber's moving limbs come apart)
        let pad = 0.35 * hExp
        var merged = true
        while merged && boxes.count > 1 {
            merged = false
            outer: for i in 0..<boxes.count {
                for j in (i + 1)..<boxes.count {
                    let a = boxes[i], b = boxes[j]
                    if a[0] - pad <= b[2] && b[0] - pad <= a[2] && a[1] - pad <= b[3] && b[1] - pad <= a[3] {
                        let nb = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                        if nb[3] - nb[1] <= 1.6 * hExp && nb[2] - nb[0] <= 1.6 * hExp {
                            boxes[i] = nb
                            boxes.remove(at: j)
                            merged = true
                            break outer
                        }
                    }
                }
            }
        }
        var out: [Box] = []
        for b in boxes {
            let bw = b[2] - b[0], bh = b[3] - b[1]
            if bh > 2.5 * hExp || bw > 2.5 * hExp || max(bw, bh) < 0.15 * hExp { continue }  // too big, or a speck
            if bh > 4 * bw && bw < 0.12 * hExp { continue }                                  // rope-like sliver
            out.append(Box(b[0], b[1], b[2], b[3]))
        }
        return out
    }

    struct Component { var x0: Int, y0: Int, x1: Int, y1: Int, area: Int }

    /// 8-connected components (cv2.connectedComponentsWithStats stats), ordered like OpenCV's
    /// block-based labelling: by the 2x2 block (row pair, then column pair) where each starts.
    static func components(_ m: [UInt8], _ w: Int, _ h: Int) -> [Component] {
        var label = [Int32](repeating: -1, count: w * h)
        var comps: [(order: Int, c: Component)] = []
        var stack: [Int] = []
        for yy in stride(from: 0, to: h, by: 2) {
            for xx in stride(from: 0, to: w, by: 2) {
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let y = yy + dy, x = xx + dx
                    guard y < h && x < w, m[y * w + x] != 0, label[y * w + x] < 0 else { continue }
                    let id = Int32(comps.count)
                    var c = Component(x0: x, y0: y, x1: x, y1: y, area: 0)
                    label[y * w + x] = id
                    stack.append(y * w + x)
                    while let p = stack.popLast() {
                        let py = p / w, px = p % w
                        c.area += 1
                        c.x0 = min(c.x0, px); c.x1 = max(c.x1, px); c.y0 = min(c.y0, py); c.y1 = max(c.y1, py)
                        for ny in max(py - 1, 0)...min(py + 1, h - 1) {
                            for nx in max(px - 1, 0)...min(px + 1, w - 1) where m[ny * w + nx] != 0 && label[ny * w + nx] < 0 {
                                label[ny * w + nx] = id
                                stack.append(ny * w + nx)
                            }
                        }
                    }
                    comps.append((comps.count, c))
                }
            }
        }
        return comps.map(\.c)
    }
}

/// np.median (mean of the two middle values for an even count).
func medianOf(_ v: [Float]) -> Float {
    if v.isEmpty { return 0 }
    let s = v.sorted()
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}

/// Binary dilation / erosion with a (2r+1)^2 square; pixels outside the image never count
/// (OpenCV's default morphology border).
func morph(_ m: [UInt8], _ w: Int, _ h: Int, radius r: Int, dilate: Bool) -> [UInt8] {
    var tmp = [UInt8](repeating: 0, count: w * h), out = tmp
    for y in 0..<h {
        for x in 0..<w {
            var v: UInt8 = dilate ? 0 : 1
            for xx in max(x - r, 0)...min(x + r, w - 1) { v = dilate ? max(v, m[y * w + xx]) : min(v, m[y * w + xx]) }
            tmp[y * w + x] = v
        }
    }
    for y in 0..<h {
        for x in 0..<w {
            var v: UInt8 = dilate ? 0 : 1
            for yy in max(y - r, 0)...min(y + r, h - 1) { v = dilate ? max(v, tmp[yy * w + x]) : min(v, tmp[yy * w + x]) }
            out[y * w + x] = v
        }
    }
    return out
}

/// cv2.GaussianBlur(plane, (0, 0), sigma) for float32: kernel size round(8 sigma + 1) | 1,
/// separable, BORDER_REFLECT_101.
func gaussianBlur(_ src: [Float], _ w: Int, _ h: Int, sigma: Float) -> [Float] {
    let n = Int((Double(sigma) * 4 * 2 + 1).rounded(.toNearestOrEven)) | 1
    let half = n / 2
    let s2 = -0.5 / (Double(sigma) * Double(sigma))
    var k = (0..<n).map { i -> Float in let x = Double(i) - Double(n - 1) * 0.5; return Float(exp(s2 * x * x)) }
    let sum = k.reduce(Float(0), +)
    k = k.map { $0 / sum }
    func reflect(_ p: Int, _ len: Int) -> Int {
        if len == 1 { return 0 }
        var p = p
        while p < 0 || p >= len { p = p < 0 ? -p : 2 * len - p - 2 }
        return p
    }
    func filterRows(_ a: [Float], _ w: Int, _ h: Int) -> [Float] {
        var out = [Float](repeating: 0, count: w * h)
        var padded = [Float](repeating: 0, count: w + 2 * half)
        for y in 0..<h {
            for i in 0..<(w + 2 * half) { padded[i] = a[y * w + reflect(i - half, w)] }
            out.withUnsafeMutableBufferPointer { o in
                padded.withUnsafeBufferPointer { p in
                    vDSP_conv(p.baseAddress!, 1, k, 1, o.baseAddress! + y * w, 1, vDSP_Length(w), vDSP_Length(n))
                }
            }
        }
        return out
    }
    func transpose(_ a: [Float], _ w: Int, _ h: Int) -> [Float] {
        var t = [Float](repeating: 0, count: w * h)
        vDSP_mtrans(a, 1, &t, 1, vDSP_Length(w), vDSP_Length(h))
        return t
    }
    let rows = filterRows(src, w, h)
    return transpose(filterRows(transpose(rows, w, h), h, w), h, w)
}
