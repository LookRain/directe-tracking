import CoreML
import CoreVideo
import Foundation
import TrackerCore

/// One person before description: box, keypoints (empty for mask-only people), mask outline.
struct Person {
    var box: SIMD4<Float>
    var kpts: [Keypoint]
    var poly: [SIMD2<Double>]?
}

@inline(__always) func toBox(_ b: SIMD4<Float>) -> Box { Box(Double(b[0]), Double(b[1]), Double(b[2]), Double(b[3])) }
@inline(__always) func toF(_ b: Box) -> SIMD4<Float> { SIMD4(Float(b.x0), Float(b.y0), Float(b.x1), Float(b.y1)) }

/// Turns a frame into tracker candidates: directe_tracking/online.py perceive(), _describe(), second_chance().
/// Not thread-safe: use one Perceiver per task.
public final class Perceiver {
    public let models: PerceptionModels
    /// Embed plain crops instead of masked ones (lab runner option `plain_reid`; needs the tracker's
    /// plain calibration reid_lo 0.47 / reid_hi 0.78).
    public var plainReID = false
    /// Run the mask model in the focus window (people the pose model missed, colour inside the body, masked re-ID
    /// crops). Off: poses only, colour from the keypoint regions, plain crops embedded by `PerceptionModels.plainReID`
    /// (the tracker needs that net's calibration, reid_lo 0.53 / reid_hi 0.72). Second chance keeps its masks for
    /// finding someone, not for describing them.
    /// Without the mask model it's off; without the plain-crop net it's on (the small set has only masked OSNet x0.25).
    public var masks = true {
        didSet {
            if models.focusSeg == nil { masks = false }
            if !masks, let p = models.plainEmbedder() {
                embedder = p
            } else {
                masks = models.focusSeg != nil
                embedder = models.reid
            }
        }
    }
    private lazy var embedder: Embedder = models.reid
    /// Model calls so far, as the lab counts them (pose, seg, reid, second).
    public private(set) var calls: [String: Int] = ["pose": 0, "seg": 0, "reid": 0, "second": 0]

    // directe_tracking/online.py constants
    public static let focus = 640.0        // FOCUS: focus window side (px), minimum
    public static let focusK = 3.0         // FOCUS_K: window side = max(FOCUS, FOCUS_K x climber height)
    public static let scK = 1.6            // SC_K: second-chance window side in predicted body heights
    public static let scMin = 213.0        // SC_MIN
    public static let scReach = 1.0        // SC_REACH
    public static let scSize = (0.5, 2.0)  // SC_SIZE

    /// Where the time goes (model passes, colour features, re-ID), for diagnostics.
    public var clock = StageClock()
    /// Run the whole-frame pass on the GPU copy of its model while `gate` says the app is in the foreground, so it
    /// overlaps the focus passes on the Neural Engine (JobOptions.gpuWide). A GPU pass that fails runs again on the
    /// Neural Engine. Slightly different numbers from the Neural Engine's.
    public var gpuWide = false
    public var gate = ComputeGate.shared
    private let wideCounts = WideCounts()
    /// Whole-frame passes run so far: on the GPU, on the Neural Engine, and GPU passes that failed (then redone on the
    /// Neural Engine).
    public var widePasses: (gpu: Int, ane: Int, gpuFailed: Int) { wideCounts.value }

    let pool = BufferPool()

    public init(models: PerceptionModels) { self.models = models }

    /// lab _window(cx, cy, side, W, H): a square window of `side` (at most the frame's short side),
    /// centred on (cx, cy) and shifted to lie inside the frame. Integer pixel coordinates.
    public static func window(cx: Double, cy: Double, side: Double, width W: Int, height H: Int) -> Box {
        let s = Int(min(side, Double(W), Double(H)))
        let x0 = Int(min(max(cx - Double(s) / 2, 0), Double(W - s)))
        let y0 = Int(min(max(cy - Double(s) / 2, 0), Double(H - s)))
        return Box(Double(x0), Double(y0), Double(x0 + s), Double(y0 + s))
    }

    /// The runner's focus window around a predicted box: side max(FOCUS, FOCUS_K x its height).
    public static func focusWindow(around pb: Box, width W: Int, height H: Int) -> Box {
        window(cx: pb.center.x, cy: pb.center.y, side: max(focus, focusK * pb.height), width: W, height: H)
    }

    // MARK: perceive

    /// lab perceive(): wide pass on the whole frame, plus (with `focus`) pose + masks at native
    /// resolution in that window, merged and described. `focus` is a window from `window(...)`.
    /// `wide: false` skips the whole-frame pass (only with a focus window): people outside the window go unseen.
    /// `ahead`: this frame's whole-frame pass, started earlier with `startWide` (same results; the Neural Engine runs it
    /// while the CPU is busy between passes).
    public func perceive(_ frame: CVPixelBuffer, focus: Box?, wide: Bool = true, ahead: WidePass? = nil) throws -> [Candidate] {
        try withPixels(frame) { view in try perceive(view, focus: focus, wide: wide, ahead: ahead) }
    }

    func perceive(_ view: PixelView, focus: Box?, wide doWide: Bool = true, ahead: WidePass? = nil) throws -> [Candidate] {
        let widePass = { () throws -> [PoseDet] in
            guard doWide || focus == nil else { return [] }
            if let ahead { return try ahead.wait() }
            return try self.clock.time("wide pose") { try self.wideDetect(view) }
        }
        guard let focus else {
            calls["pose", default: 0] += 1
            return try describe(view, try widePass().map { Person(box: $0.box, kpts: $0.kpts, poly: nil) })
        }
        let x0 = Int(focus.x0), y0 = Int(focus.y0), x1 = Int(focus.x1), y1 = Int(focus.y1)
        let crop = view.crop(x0, y0, x1 - x0, y1 - y0)
        // The three passes don't depend on each other: run them together, so one's CPU work (letterbox, NMS, masks)
        // overlaps another's time on the Neural Engine. Same results as one after another.
        let masks = self.masks && models.focusSeg != nil
        let (wide, (fp, fs)) = try concurrently(widePass) {
            let canvas = try clock.time("focus canvas") { try Canvas(crop, pool: pool) }  // the lab letterboxes the window once per model; same pixels
            return try concurrently({ try self.clock.time("focus pose") { try self.models.focusPose.detect(canvas) } },
                                    { masks ? try self.clock.time("focus seg") { try self.models.focusSeg!.detect(canvas) } : [] })
        }
        calls["pose", default: 0] += doWide ? 2 : 1
        if masks { calls["seg", default: 0] += 1 }
        let off = SIMD4<Float>(Float(x0), Float(y0), Float(x0), Float(y0))
        let poses = fp.map { d in
            PoseDet(box: d.box + off, conf: d.conf,
                    kpts: d.kpts.map { Keypoint(x: $0.x + Float(x0), y: $0.y + Float(y0), conf: $0.conf) })
        }
        let segBoxes = fs.map { $0.box + off }
        let segPolys = fs.map { d in d.poly.map { SIMD2<Double>(Double($0.x) + Double(x0), Double($0.y) + Double(y0)) } }
        var people = Self.merge(poses, segBoxes, segPolys)
        // wide-pass people not already seen in the focus window
        for d in wide {
            let b = d.box
            let inside = b[0] >= Float(x0) && b[1] >= Float(y0) && b[2] <= Float(x1) && b[3] <= Float(y1)
            if !inside && !people.contains(where: { toBox(b).samePerson(toBox($0.box)) }) {
                people.append(Person(box: b, kpts: d.kpts, poly: nil))
            }
        }
        return try describe(view, people)
    }

    /// The whole-frame pass: on the GPU copy when asked for and allowed, else (or if that fails) the Neural Engine.
    func wideDetect(_ view: PixelView) throws -> [PoseDet] {
        if gpuWide, gate.foreground, let net = models.gpuWidePose() {
            do {
                let d = try net.detect(view, pool: pool)
                wideCounts.add(gpu: 1)
                return d
            } catch {
                wideCounts.add(failed: 1)
            }
        }
        let d = try models.widePose.detect(view, pool: pool)
        wideCounts.add(ane: 1)
        return d
    }

    /// Pose boxes take the mask they overlap best (IoU >= 0.5); masks no pose box claims and that
    /// aren't the same person as one (containment, IoU 0.3) join as mask-only people.
    static func merge(_ poses: [PoseDet], _ segBoxes: [SIMD4<Float>], _ segPolys: [[SIMD2<Double>]]) -> [Person] {
        var people: [Person] = []
        var used = Set<Int>()
        for d in poses {
            var poly: [SIMD2<Double>]?
            if !segBoxes.isEmpty {
                var j = 0, best = -Double.infinity
                for (k, s) in segBoxes.enumerated() {
                    let ov = toBox(d.box).iou(toBox(s))
                    if ov > best { best = ov; j = k }
                }
                if best >= 0.5 { poly = segPolys[j]; used.insert(j) }
            }
            people.append(Person(box: d.box, kpts: d.kpts, poly: poly))
        }
        for j in 0..<segBoxes.count where !used.contains(j) {
            if poses.isEmpty || !poses.contains(where: { toBox(segBoxes[j]).samePerson(toBox($0.box), iouThreshold: 0.3) }) {
                people.append(Person(box: segBoxes[j], kpts: [], poly: segPolys[j]))
            }
        }
        return people
    }

    /// lab _describe(): colour inside masks and re-ID embeddings.
    func describe(_ view: PixelView, _ people: [Person]) throws -> [Candidate] {
        if people.isEmpty { return [] }
        var described = [(Candidate, BGRAImage)?](repeating: nil, count: people.count)
        let masked = !plainReID && masks
        let bodies = masks  // without masks in the focus window, none anywhere: colour as the profile was made
        clock.time("colour features") {
            // each person on its own core (same results, in the same order)
            described.withUnsafeMutableBufferPointer { buf in
                let d = buf
                let one = { (i: Int) in
                    let p = people[i]
                    let (f, v) = Appearance.features(view, box: p.box, kpts: p.kpts, body: bodies ? p.poly : nil)
                    d[i] = (Candidate(box: toBox(p.box), feats: f, valids: v, emb: nil, keypoints: p.kpts),
                            Embedder.crop(view, box: p.box, poly: p.poly, masked: masked))
                }
                if people.count == 1 || !PixelWork.multicore { for i in people.indices { one(i) } } else {
                    DispatchQueue.concurrentPerform(iterations: people.count, execute: one)
                }
            }
        }
        var out = described.map { $0!.0 }
        let crops = described.map { $0!.1 }
        let embs = try clock.time("re-id") { try embedder.embed(crops, pool: pool) }
        calls["reid", default: 0] += people.count
        for i in out.indices { out[i].emb = embs[i] }
        return out
    }

    // MARK: second chance

    /// lab second_chance(): nobody at the prediction `pb`, so look again at ~2x (pose + masks) in a
    /// window of 1.6 body heights around it and add at most one person there: centred within one
    /// predicted body height, 0.5-2x as tall, not already a candidate, and (with `motion`) where the
    /// wall moved. Returns `candidates` unchanged when someone is already at the prediction.
    public func secondChance(_ frame: CVPixelBuffer, predicted pb: Box, candidates c: [Candidate],
                             motion: MotionEvidence? = nil) throws -> [Candidate] {
        try withPixels(frame) { view in try secondChance(view, predicted: pb, candidates: c, motion: motion) }
    }

    func secondChance(_ view: PixelView, predicted pbD: Box, candidates c: [Candidate], motion: MotionEvidence?) throws -> [Candidate] {
        let W = view.width, H = view.height
        let pb = toF(pbD)
        let h = pb[3] - pb[1]
        if h <= 0 || c.contains(where: { toBox(pb).iou($0.box) >= 0.3 }) { return c }
        let side = min(max(Self.scK * Double(h), Self.scMin), Double(min(W, H)))
        let win = Self.window(cx: Double((pb[0] + pb[2]) / 2), cy: Double((pb[1] + pb[3]) / 2), side: side, width: W, height: H)
        let x0 = Int(win.x0), y0 = Int(win.y0), x1 = Int(win.x1), y1 = Int(win.y1)
        let s = 640.0 / Double(y1 - y0)
        let up = BGRAImage(width: 640, height: 640)
        let canvas = try clock.time("second canvas") {
            cvResize(view.crop(x0, y0, x1 - x0, y1 - y0), up.view, s > 1 ? .linear : .area)
            return try Canvas(up.view, pool: pool)
        }
        let (fp, fs) = try concurrently({ try self.clock.time("second pose") { try self.models.secondPose.detect(canvas) } },
                                        { try self.clock.time("second seg") { try self.models.secondSeg.detect(canvas) } })
        calls["pose", default: 0] += 1
        calls["seg", default: 0] += 1
        calls["second", default: 0] += 1
        let sf = Float(s)
        let off = SIMD4<Float>(Float(x0), Float(y0), Float(x0), Float(y0))
        let poses = fp.map { d in
            PoseDet(box: d.box / sf + off, conf: d.conf,
                    kpts: d.kpts.map { Keypoint(x: $0.x / sf + Float(x0), y: $0.y / sf + Float(y0), conf: $0.conf) })
        }
        let segBoxes = fs.map { $0.box / sf + off }
        let segPolys = fs.map { d in d.poly.map { SIMD2<Double>(Double($0.x / sf) + Double(x0), Double($0.y / sf) + Double(y0)) } }
        let people = Self.merge(poses, segBoxes, segPolys)
        if people.isEmpty { return c }
        let pc = SIMD2<Float>((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2)
        var best: Int?
        var bestD = Float.infinity
        for (i, p) in people.enumerated() {
            let b = p.box
            let ctr: SIMD2<Float> = SIMD2(b[0] + b[2], b[1] + b[3]) / 2
            let dv: SIMD2<Float> = ctr - pc
            let d: Float = (dv.x * dv.x + dv.y * dv.y).squareRoot() / h
            let r = (b[3] - b[1]) / h
            var ok = d <= Float(Self.scReach) && r >= Float(Self.scSize.0) && r <= Float(Self.scSize.1)
            if ok, let motion { ok = motion.supports(toBox(b)) }
            if ok && !c.isEmpty { ok = !c.contains(where: { toBox(b).samePerson($0.box) }) }
            if ok && d < bestD { bestD = d; best = i }
        }
        guard let best else { return c }
        return c + (try describe(view, [people[best]]))
    }

    // MARK: tap screen

    /// The outline of the person in `box` (source pixels), from the mask model on a window around them, for the
    /// tap screen's subject highlight. Nil when the mask model finds no one there.
    public func outline(of box: Box, in frame: CVPixelBuffer) throws -> [SIMD2<Double>]? {
        try withPixels(frame) { view in
            let win = Self.window(cx: box.center.x, cy: box.center.y, side: max(Self.focus, 1.4 * max(box.height, box.width)),
                                  width: view.width, height: view.height)
            let x0 = Int(win.x0), y0 = Int(win.y0)
            let canvas = try Canvas(view.crop(x0, y0, Int(win.x1) - x0, Int(win.y1) - y0), pool: pool)
            let found = try (models.focusSeg ?? models.secondSeg).detect(canvas)  // the small model when the medium isn't there
            let off = SIMD4<Float>(Float(x0), Float(y0), Float(x0), Float(y0))
            guard let best = found.max(by: { toBox($0.box + off).iou(box) < toBox($1.box + off).iou(box) }),
                  toBox(best.box + off).iou(box) >= 0.3, best.poly.count >= 3 else { return nil }
            return best.poly.map { SIMD2(Double($0.x) + Double(x0), Double($0.y) + Double(y0)) }
        }
    }

    /// Everyone in a frame, for the tap screen: the wide pass plus (with `tiled`) the pose model on
    /// overlapping square tiles of about native resolution, so small climbers far up the wall get a
    /// box too. Tile people the wide pass already has (same person, IoU 0.3) are dropped. With
    /// `describe`, candidates carry colour features (no masks) and embeddings; otherwise only boxes
    /// and keypoints (feats zero, valids false, emb nil).
    public func people(in frame: CVPixelBuffer, tiled: Bool = true, describe: Bool = false) throws -> [Candidate] {
        try withPixels(frame) { view in
            var dets = try models.widePose.detect(view, pool: pool)
            calls["pose", default: 0] += 1
            if tiled {
                let W = view.width, H = view.height
                let side = min(max(640, min(W, H) / 2), min(W, H))
                var found: [PoseDet] = []
                for ty in Self.tiles(H, side) {
                    for tx in Self.tiles(W, side) {
                        let t = try models.focusPose.detect(view.crop(tx, ty, side, side), pool: pool)
                        calls["pose", default: 0] += 1
                        for d in t {
                            let off = SIMD4<Float>(Float(tx), Float(ty), Float(tx), Float(ty))
                            let b = d.box + off
                            // a box touching an inner tile edge is a cut-off person; another tile has them whole
                            let cut = (b[0] <= Float(tx) + 1 && tx > 0) || (b[1] <= Float(ty) + 1 && ty > 0)
                                || (b[2] >= Float(tx + side) - 1 && tx + side < W) || (b[3] >= Float(ty + side) - 1 && ty + side < H)
                            if cut { continue }
                            found.append(PoseDet(box: b, conf: d.conf, kpts: d.kpts.map {
                                Keypoint(x: $0.x + Float(tx), y: $0.y + Float(ty), conf: $0.conf) }))
                        }
                    }
                }
                for d in found.sorted(by: { $0.conf > $1.conf })
                where !dets.contains(where: { toBox(d.box).samePerson(toBox($0.box), iouThreshold: 0.3) }) {
                    dets.append(d)
                }
            }
            let people = dets.map { Person(box: $0.box, kpts: $0.kpts, poly: nil) }
            if describe { return try self.describe(view, people) }
            return people.map {
                Candidate(box: toBox($0.box), feats: [[Float]](repeating: [Float](repeating: 0, count: AppearanceSpec.dim),
                                                              count: AppearanceSpec.regions.count),
                          valids: [Bool](repeating: false, count: AppearanceSpec.regions.count), emb: nil, keypoints: $0.kpts)
            }
        }
    }

    /// Tile origins covering [0, length) with tiles of `side` overlapping by at least 20%.
    static func tiles(_ length: Int, _ side: Int) -> [Int] {
        if length <= side { return [0] }
        let n = Int((Double(length - side) / (0.8 * Double(side))).rounded(.up)) + 1
        return (0..<n).map { Int((Double(length - side) * Double($0) / Double(n - 1)).rounded()) }
    }
}

/// The whole-frame pass for one frame, run on a worker ahead of `perceive` (`Perceiver.startWide`).
public final class WidePass: @unchecked Sendable {
    public let frame: CVPixelBuffer
    private let done = DispatchGroup()
    private var result: Result<[PoseDet], Error>?

    init(frame: CVPixelBuffer) { self.frame = frame; done.enter() }

    fileprivate func finish(_ r: Result<[PoseDet], Error>) { result = r; done.leave() }

    /// Waits for it to finish.
    func wait() throws -> [PoseDet] {
        done.wait()
        return try result!.get()
    }
}

extension Perceiver {
    private static let aheadQueue = DispatchQueue(label: "climbcam.wide-ahead", qos: .userInitiated)

    /// Starts the whole-frame pass on `frame` on a worker; hand it to `perceive(_:focus:wide:ahead:)` for that frame.
    /// Thread-safe with respect to the Perceiver's own work (the pass only reads the models and the buffer pool).
    public func startWide(_ frame: CVPixelBuffer) -> WidePass {
        let w = WidePass(frame: frame)
        Self.aheadQueue.async {
            w.finish(Result {
                try autoreleasepool {
                    try self.clock.time("wide pose") { try withPixels(frame) { try self.wideDetect($0) } }
                }
            })
        }
        return w
    }
}

/// Whole-frame pass counts, updated from the job thread and the wide-ahead worker.
final class WideCounts: @unchecked Sendable {
    private let lock = NSLock()
    private var gpu = 0, ane = 0, failed = 0
    func add(gpu g: Int = 0, ane a: Int = 0, failed f: Int = 0) { lock.withLock { gpu += g; ane += a; failed += f } }
    var value: (gpu: Int, ane: Int, gpuFailed: Int) { lock.withLock { (gpu, ane, failed) } }
}

/// Runs `a` and `b` at the same time (this thread and a worker) and returns both, or the first error.
public func concurrently<A, B>(_ a: () throws -> A, _ b: () throws -> B) throws -> (A, B) {
    var ra: Result<A, Error>?
    var rb: Result<B, Error>?
    DispatchQueue.concurrentPerform(iterations: 2) { i in
        if i == 0 { ra = Result { try a() } } else { rb = Result { try b() } }
    }
    return (try ra!.get(), try rb!.get())
}
