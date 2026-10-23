import Accelerate
import CoreML
import CoreVideo
import Foundation
import TrackerCore

// YOLO26 pose and segmentation as Ultralytics' predictor runs them in the lab (directe_tracking/detect.py,
// directe_tracking/identity.py Segmenter): LetterBox (rect, stride 32, grey 114, cv2 INTER_LINEAR), the
// one-to-many head + non_max_suppression (best class > conf, class 0, greedy IoU 0.7, max 300; the
// lab's predictor default nms=None, not the NMS-free one-to-one head), scale_boxes / scale_coords
// back to the source, masks via process_mask (bilinear upsample of the prototype logits to the
// network input, > 0, cropped to the box) and Masks.xy (masks2segments + scale_coords).

struct PoseDet {
    var box: SIMD4<Float>       // source pixels, x0 y0 x1 y1
    var conf: Float
    var kpts: [Keypoint]        // 17, source pixels
}

struct SegDet {
    var box: SIMD4<Float>
    var conf: Float
    var poly: [SIMD2<Float>]    // mask outline, source pixels (float32 like the lab's r.masks.xy)
}

/// Ultralytics LetterBox geometry for a source of srcW x srcH into a `size` canvas.
struct Letterbox {
    let newW: Int, newH: Int, left: Int, top: Int, canvasW: Int, canvasH: Int

    init(srcW: Int, srcH: Int, size: Int = 640, auto: Bool, stride: Int = 32) {
        let r = min(Double(size) / Double(srcH), Double(size) / Double(srcW))
        newW = max(pyRound(Double(srcW) * r), 1)
        newH = max(pyRound(Double(srcH) * r), 1)
        var dw = size - newW, dh = size - newH
        if auto { dw %= stride; dh %= stride }
        let hw = Double(dw) / 2, hh = Double(dh) / 2
        top = pyRound(hh - 0.1); left = pyRound(hw - 0.1)
        canvasW = newW + left + pyRound(hw + 0.1)
        canvasH = newH + top + pyRound(hh + 0.1)
    }
}

/// ops.scale_boxes / scale_coords from an img1 (network canvas, mask) shape back to img0 (source).
struct ScaleBack {
    let gx: Float, gy: Float, px: Float, py: Float, w: Float, h: Float

    init(img1W: Int, img1H: Int, img0W: Int, img0H: Int) {
        let gain = min(Double(img1H) / Double(img0H), Double(img1W) / Double(img0W))
        let nh = max(pyRound(Double(img0H) * gain), 1), nw = max(pyRound(Double(img0W) * gain), 1)
        gy = Float(Double(nh) / Double(img0H)); gx = Float(Double(nw) / Double(img0W))
        px = Float(pyRound(Double(img1W - nw) / 2 - 0.1)); py = Float(pyRound(Double(img1H - nh) / 2 - 0.1))
        w = Float(img0W); h = Float(img0H)
    }

    @inline(__always) func x(_ v: Float) -> Float { min(max((v - px) / gx, 0), w) }
    @inline(__always) func y(_ v: Float) -> Float { min(max((v - py) / gy, 0), h) }
    func box(_ b: SIMD4<Float>) -> SIMD4<Float> { SIMD4(x(b[0]), y(b[1]), x(b[2]), y(b[3])) }
}

/// Whether a Core ML array's values are stored densely in row-major order.
func isDense(_ a: MLMultiArray) -> Bool {
    let shape = a.shape.map { $0.intValue }, strides = a.strides.map { $0.intValue }
    var expect = 1
    for d in stride(from: shape.count - 1, through: 0, by: -1) {
        if shape[d] > 1 && strides[d] != expect { return false }
        expect *= shape[d]
    }
    return true
}

/// `body` with a Core ML output's values as dense row-major floats: the array's own memory when it already is (dense
/// float32, the usual case: no copy of the seg model's 7 MB of outputs), else a dense copy.
func withDenseFloats<R>(_ a: MLMultiArray, _ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
    if a.dataType == .float32 && isDense(a) {
        return try withExtendedLifetime(a) {
            try body(UnsafeBufferPointer(start: a.dataPointer.assumingMemoryBound(to: Float.self), count: a.count))
        }
    }
    return try denseFloats(a).withUnsafeBufferPointer(body)
}

/// A Core ML output as a dense [Float] in row-major order (respects the array's strides).
func denseFloats(_ a: MLMultiArray) -> [Float] {
    let shape = a.shape.map { $0.intValue }, strides = a.strides.map { $0.intValue }
    let n = shape.reduce(1, *)
    var out = [Float](repeating: 0, count: n)
    let rank = shape.count
    if isDense(a) && a.dataType == .float32 {
        out.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(start: a.dataPointer, count: n * 4)) }
        return out
    }
    let inner = shape[rank - 1], innerStride = strides[rank - 1]
    let outer = n / max(inner, 1)
    func offset(_ k: Int) -> Int {
        var k = k, o = 0
        for d in stride(from: rank - 2, through: 0, by: -1) { o += (k % shape[d]) * strides[d]; k /= shape[d] }
        return o
    }
    switch a.dataType {
    case .float32:
        let p = a.dataPointer.assumingMemoryBound(to: Float.self)
        out.withUnsafeMutableBufferPointer { dst in
            for k in 0..<outer {
                let o = offset(k)
                for i in 0..<inner { dst[k * inner + i] = p[o + i * innerStride] }
            }
        }
    case .float16:
        let p = a.dataPointer.assumingMemoryBound(to: UInt16.self)
        var bits = [UInt16](repeating: 0, count: n)
        for k in 0..<outer {
            let o = offset(k)
            for i in 0..<inner { bits[k * inner + i] = p[o + i * innerStride] }
        }
        bits.withUnsafeMutableBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                var s = vImage_Buffer(data: src.baseAddress, height: 1, width: vImagePixelCount(n), rowBytes: n * 2)
                var d = vImage_Buffer(data: dst.baseAddress, height: 1, width: vImagePixelCount(n), rowBytes: n * 4)
                vImageConvert_Planar16FtoPlanarF(&s, &d, 0)
            }
        }
    default:
        for i in 0..<n { out[i] = a[i].floatValue }
    }
    return out
}

/// Fills a model input buffer the way LetterBox does: grey 114, the source resized at (left, top),
/// optionally placed at (ox, oy) inside a larger model input.
private func letterboxInput(_ src: PixelView, _ lb: Letterbox, into pb: CVPixelBuffer, ox: Int, oy: Int) throws {
    try withPixels(pb, readOnly: false) { dst in
        var grey: UInt32 = 0xFF72_7272  // B G R = 114, A = 255 (little-endian BGRA)
        for y in 0..<dst.height { memset_pattern4(dst.row(y), &grey, dst.width * 4) }
        cvResize(src, dst.crop(ox + lb.left, oy + lb.top, lb.newW, lb.newH), .linear)
    }
}

/// One NMS survivor: anchor index, xyxy box in network pixels, score.
struct RawDet { var anchor: Int; var box: SIMD4<Float>; var conf: Float }

/// ultralytics.utils.nms.non_max_suppression for one image of a [C, N] head output (channel-major):
/// rows 0-3 cx cy w h, 4..<4+nc class scores; best class only, conf > `conf`, class 0, greedy NMS
/// (torchvision: suppress IoU > `iou`) by descending score, at most `maxDet`.
func nonMaxSuppression(_ p: UnsafeBufferPointer<Float>, channels: Int, anchors n: Int, nc: Int, conf: Float,
                       iou: Float = 0.7, maxDet: Int = 300) -> [RawDet] {
    var cands: [RawDet] = []
    for a in 0..<n {
        // class 0 must be the best class (first index wins ties) and above conf
        let best = p[4 * n + a]
        guard best > conf else { continue }
        var other = false
        for k in 1..<max(nc, 1) where p[(4 + k) * n + a] > best { other = true; break }
        if other { continue }
        let cx = p[a], cy = p[n + a], hw = p[2 * n + a] / 2, hh = p[3 * n + a] / 2
        cands.append(RawDet(anchor: a, box: SIMD4(cx - hw, cy - hh, cx + hw, cy + hh), conf: best))
    }
    cands.sort { $0.conf > $1.conf }
    var keep: [RawDet] = []
    var removed = [Bool](repeating: false, count: cands.count)
    for i in cands.indices where !removed[i] {
        keep.append(cands[i])
        if keep.count == maxDet { break }
        let a = cands[i].box
        let areaA = (a[2] - a[0]) * (a[3] - a[1])
        for k in (i + 1)..<cands.count where !removed[k] {
            let b = cands[k].box
            let iw = max(min(a[2], b[2]) - max(a[0], b[0]), 0), ih = max(min(a[3], b[3]) - max(a[1], b[1]), 0)
            let inter = iw * ih
            if inter / (areaA + (b[2] - b[0]) * (b[3] - b[1]) - inter) > iou { removed[k] = true }
        }
    }
    return keep
}

private func predict(_ model: MLModel, _ pb: CVPixelBuffer) throws -> MLFeatureProvider {
    try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pb)]))
}

/// A 640x640 network input, letterboxed from a source the way Ultralytics does it (rect for a square
/// source, so no padding; the lab only runs the 640 models on square windows, other shapes are padded).
struct Canvas {
    let pb: CVPixelBuffer
    let lb: Letterbox
    let srcW: Int, srcH: Int

    init(_ src: PixelView, pool: BufferPool) throws {
        lb = Letterbox(srcW: src.width, srcH: src.height, auto: src.width == src.height)
        pb = try pool.make(width: 640, height: 640)
        srcW = src.width; srcH = src.height
        try letterboxInput(src, lb, into: pb, ox: 0, oy: 0)
    }
}

final class PoseNet {
    let models: [SIMD2<Int>: MLModel]   // (H, W) -> model
    var conf: Float = 0.25               // lab Detector default

    init(models: [SIMD2<Int>: MLModel]) { self.models = models }

    /// Detector()(src): pose on the whole of `src` at imgsz 640 (rect letterbox).
    func detect(_ src: PixelView, pool: BufferPool) throws -> [PoseDet] {
        let lb = Letterbox(srcW: src.width, srcH: src.height, auto: true)
        var model = models[SIMD2(lb.canvasH, lb.canvasW)]
        var mw = lb.canvasW, mh = lb.canvasH, ox = 0, oy = 0
        if model == nil {
            // No model for this rectangle: the square model with the canvas placed on the stride-32
            // grid (same anchors for the image content; only the padding differs).
            model = models[SIMD2(640, 640)]!
            mw = 640; mh = 640
            ox = (640 - lb.canvasW) / 2 / 32 * 32; oy = (640 - lb.canvasH) / 2 / 32 * 32
        }
        let pb = try pool.make(width: mw, height: mh)
        try letterboxInput(src, lb, into: pb, ox: ox, oy: oy)
        return try decode(model!, pb, lb, ox: ox, oy: oy, srcW: src.width, srcH: src.height)
    }

    /// Pose on a prepared 640x640 square canvas (shared with the mask model).
    func detect(_ c: Canvas) throws -> [PoseDet] {
        try decode(models[SIMD2(640, 640)]!, c.pb, c.lb, ox: 0, oy: 0, srcW: c.srcW, srcH: c.srcH)
    }

    private func decode(_ model: MLModel, _ pb: CVPixelBuffer, _ lb: Letterbox, ox: Int, oy: Int, srcW: Int, srcH: Int) throws -> [PoseDet] {
        guard let out = try predict(model, pb).featureValue(for: "detections")?.multiArrayValue else {
            throw PerceptionError.modelOutput("pose detections")
        }
        let ch = out.shape[1].intValue, n = out.shape[2].intValue
        let sb = ScaleBack(img1W: lb.canvasW, img1H: lb.canvasH, img0W: srcW, img0H: srcH)
        let fx = Float(ox), fy = Float(oy)
        let nk = (ch - 5) / 3
        return withDenseFloats(out) { p in
            var dets: [PoseDet] = []
            for d in nonMaxSuppression(p, channels: ch, anchors: n, nc: 1, conf: conf) {
                let b = d.box
                let box = sb.box(SIMD4(b[0] - fx, b[1] - fy, b[2] - fx, b[3] - fy))
                var k: [Keypoint] = []
                k.reserveCapacity(nk)
                for j in 0..<nk {
                    let c = 5 + 3 * j
                    k.append(Keypoint(x: sb.x(p[c * n + d.anchor] - fx), y: sb.y(p[(c + 1) * n + d.anchor] - fy),
                                      conf: p[(c + 2) * n + d.anchor]))
                }
                dets.append(PoseDet(box: box, conf: d.conf, kpts: k))
            }
            return dets
        }
    }
}

final class SegNet {
    let model: MLModel
    var conf: Float = 0.2                // lab Segmenter

    init(model: MLModel) { self.model = model }

    /// Segmenter()(src): person boxes and mask outlines (retina_masks=False).
    func detect(_ src: PixelView, pool: BufferPool) throws -> [SegDet] {
        try detect(try Canvas(src, pool: pool))
    }

    func detect(_ c: Canvas) throws -> [SegDet] {
        let lb = c.lb
        let out = try predict(model, c.pb)
        guard let det = out.featureValue(for: "detections")?.multiArrayValue,
              let proto = out.featureValue(for: "protos")?.multiArrayValue else {
            throw PerceptionError.modelOutput("seg detections/protos")
        }
        let ch = det.shape[1].intValue, n = det.shape[2].intValue
        let pshape = proto.shape.map { $0.intValue }
        let nm = pshape[1], mh = pshape[2], mw = pshape[3]
        let nc = ch - 4 - nm
        let sb = ScaleBack(img1W: lb.canvasW, img1H: lb.canvasH, img0W: c.srcW, img0H: c.srcH)
        let (kept, coeffs) = withDenseFloats(det) { p -> ([RawDet], [[Float]]) in
            let kept = nonMaxSuppression(p, channels: ch, anchors: n, nc: nc, conf: conf)
            return (kept, kept.map { d in (0..<nm).map { p[(4 + nc + $0) * n + d.anchor] } })
        }
        if kept.isEmpty { return [] }
        // each person's mask and outline on its own core (same results, in the same order)
        var found = [SegDet?](repeating: nil, count: kept.count)
        withDenseFloats(proto) { protos in
            found.withUnsafeMutableBufferPointer { buf in
                let out = buf
                let one = { (i: Int) in
                    let d = kept[i]
                    guard let (mask, mx, my) = Self.mask(protos, nm, mw, mh, coeffs[i], d.box, lb.canvasW, lb.canvasH) else { return }
                    let pts = maskSegment(mask)
                    let poly = pts.map { SIMD2(sb.x(Float($0.x + mx)), sb.y(Float($0.y + my))) }
                    out[i] = SegDet(box: sb.box(d.box), conf: d.conf, poly: poly)
                }
                if kept.count == 1 || !PixelWork.multicore { for i in kept.indices { one(i) } } else {
                    DispatchQueue.concurrentPerform(iterations: kept.count, execute: one)
                }
            }
        }
        return found.compactMap { $0 }
    }

    /// process_mask(upsample=True) for one detection, restricted to its box (crop_mask zeroes the
    /// rest): returns the mask of the box region and its origin, or nil if the mask is empty
    /// (the predictor drops those detections).
    static func mask(_ protos: UnsafeBufferPointer<Float>, _ nm: Int, _ mw: Int, _ mh: Int, _ coeffs: [Float],
                     _ b: SIMD4<Float>, _ W: Int, _ H: Int) -> (Mask, Int, Int)? {
        let c0 = max(0, Int(b[0].rounded(.up))), c1 = min(W, Int(b[2].rounded(.up)))
        let r0 = max(0, Int(b[1].rounded(.up))), r1 = min(H, Int(b[3].rounded(.up)))
        guard c1 > c0 && r1 > r0 else { return nil }
        let sx = Float(mw) / Float(W), sy = Float(mh) / Float(H)
        func src(_ d: Int, _ s: Float) -> Float { max((Float(d) + 0.5) * s - 0.5, 0) }
        // prototype cells needed
        let px0 = Int(src(c0, sx)), px1 = min(Int(src(c1 - 1, sx)) + 1, mw - 1)
        let py0 = Int(src(r0, sy)), py1 = min(Int(src(r1 - 1, sy)) + 1, mh - 1)
        let lw = px1 - px0 + 1, lh = py1 - py0 + 1
        var logits = [Float](repeating: 0, count: lw * lh)
        let plane = mw * mh
        for k in 0..<nm {
            let ck = coeffs[k]
            let base = k * plane
            for y in 0..<lh {
                let o = base + (py0 + y) * mw + px0
                for x in 0..<lw { logits[y * lw + x] += ck * protos[o + x] }
            }
        }
        var m = Mask(width: c1 - c0, height: r1 - r0)
        var any = false
        for r in r0..<r1 {
            let fy = src(r, sy)
            let y0 = Int(fy), y1 = min(y0 + 1, mh - 1)
            let ly = fy - Float(y0), hy = 1 - ly
            let a = (y0 - py0) * lw, bb = (y1 - py0) * lw
            for c in c0..<c1 {
                let fx = src(c, sx)
                let x0 = Int(fx), x1 = min(x0 + 1, mw - 1)
                let lx = fx - Float(x0), hx = 1 - lx
                let v = hy * (hx * logits[a + x0 - px0] + lx * logits[a + x1 - px0])
                      + ly * (hx * logits[bb + x0 - px0] + lx * logits[bb + x1 - px0])
                if v > 0 { m[c - c0, r - r0] = 1; any = true }
            }
        }
        return any ? (m, c0, r0) : nil
    }
}
