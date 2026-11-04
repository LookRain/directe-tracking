import CoreML
import Foundation

/// Which Core ML models run in each pass. Names are weight stems; the files are compiled models
/// named `<stem>_640.mlmodelc` (640x640 input), `<stem>_640x384.mlmodelc` / `<stem>_384x640.mlmodelc`
/// (H x W, the wide pass on 16:9 portrait / landscape frames, optional) and `<reid>.mlmodelc`
/// (tools/export_coreml.py). A pose model's shapes can instead be one multifunction model,
/// `<stem>_shapes.mlmodelc` (functions `s<H>x<W>`, the weights stored once), which is preferred. A mask model named
/// `<stem>:<function>` is that function of `<stem>_shapes.mlmodelc`.
public struct ModelSet: Sendable, Hashable, CustomStringConvertible {
    public var name: String
    /// Wide pass: the whole frame letterboxed to 640 (rectangular input when the model exists).
    public var widePose: String
    /// Focus pass: the window around the climber at native resolution, pose + masks.
    public var focusPose: String
    public var focusSeg: String
    /// Second chance (2x around the prediction).
    public var secondPose: String
    public var secondSeg: String
    /// Re-ID embedder (OSNet, 256x128).
    public var reid: String
    /// Re-ID for plain crops (masks off): `PerceptionModels.plainReID` unless the set has its own.
    public var plainReid: String

    public init(name: String, widePose: String, focusPose: String, focusSeg: String,
                secondPose: String, secondSeg: String, reid: String, plainReid: String = PerceptionModels.plainReID) {
        self.name = name; self.widePose = widePose; self.focusPose = focusPose; self.focusSeg = focusSeg
        self.secondPose = secondPose; self.secondSeg = secondSeg; self.reid = reid; self.plainReid = plainReid
    }

    /// The small set, what an app bundles: the default set's detection and outline models with 8-bit weights (owner,
    /// 4 Nov: "ship the 8-bit model into the app, everything including the detection, reid … and outline models";
    /// tools/export_open_models.py): RTMO-m pose and its box finder, RTMDet-Ins-m outlines, and the 16-bit OSNets (small
    /// enough: 9 MB). The same settings as `mid`, which a download replaces it with on Wi-Fi. One-function models only:
    /// on an iPhone (iOS 27.2) the 8-bit multifunction model failed to load for the Neural Engine ("`.functionName`
    /// must be nil unless the model type is ML Program"; fine on the CPU and on a Mac, build 112), so RTMO-m comes at
    /// 640 x 640 only and second chance's mask model is the outline model.
    public static let lite = ModelSet(name: "lite", widePose: "rtmo-m-q8", focusPose: "rtmo-m-q8",
                                      focusSeg: "rtmdet-ins-m-q8", secondPose: "rtmo-m-q8", secondSeg: "rtmdet-ins-m-q8",
                                      reid: "osnet_ain_x1_0_msmt17")
    /// The app default: RTMO-m pose in the wide and focus passes, RTMO-m's box finder (function seg640 of its model: no
    /// extra weights) and RTMO-m pose again for second chance (the same loaded model), RTMDet-Ins-m outlines in the focus window when masks are on
    /// (optional: an app may leave it out), OSNet-AIN re-ID. Live segments (Dual Lens) 97.4% right, 0% someone else vs
    /// 93.4% / 0.7% for the YOLO26 set (`yoloMid`); lock-on bench 84.8% vs 84.2%. RTMO-m runs 6.9 ms at 640 x 640 on
    /// the Neural Engine vs 10.9 for yolo26m-pose (docs/review/deep-dives/safe-models.md in the app repo).
    public static let mid = ModelSet(name: "mid", widePose: "rtmo-m", focusPose: "rtmo-m",
                                     focusSeg: "rtmdet-ins-m", secondPose: "rtmo-m", secondSeg: "rtmo-m:seg640",
                                     reid: "osnet_ain_x1_0_msmt17")
    /// The RTMO-s small set the 8-bit one replaced (masks off, OSNet x0.25), for comparison.
    public static let rtmoSmall = ModelSet(name: "rtmoSmall", widePose: "rtmo-s", focusPose: "rtmo-s",
                                           focusSeg: "none", secondPose: "rtmo-s", secondSeg: "rtmo-s:seg640",
                                           reid: "osnet_x0_25_msmt17")
    /// The YOLO26 sets these replace (Ultralytics, AGPL-3.0 weights; tools/export_coreml.py), for comparison:
    /// yolo26n pose + seg at 640, OSNet x0.25 ...
    public static let yoloLite = ModelSet(name: "yoloLite", widePose: "yolo26n-pose", focusPose: "yolo26n-pose",
                                          focusSeg: "yolo26n-seg", secondPose: "yolo26n-pose", secondSeg: "yolo26n-seg",
                                          reid: "osnet_x0_25_msmt17")
    /// ... and yolo26m pose + seg at 640 in the wide and focus passes (docs/review/deep-dives/detector-size.md), the
    /// nano models for second chance, OSNet-AIN re-ID.
    public static let yoloMid = ModelSet(name: "yoloMid", widePose: "yolo26m-pose", focusPose: "yolo26m-pose",
                                         focusSeg: "yolo26m-seg", secondPose: "yolo26n-pose", secondSeg: "yolo26n-seg",
                                         reid: "osnet_ain_x1_0_msmt17")
    public static let `default` = mid

    /// The compiled models (`.mlmodelc` names) this set can't load without, each as the names any one of which will do
    /// (a pose model as one multifunction `_shapes` file or per-shape `_640` ones). The focus window's mask model and
    /// the plain-crop re-ID net are optional, so not here. An app checks its bundle with this rather than a file name of
    /// its own (build 114 looked for a renamed file, didn't find it, and ran its stand-in engine).
    public var needs: [[String]] {
        let poses = Set([widePose, focusPose, secondPose]).sorted().map { ["\($0)_shapes", "\($0)_640"] }
        let seg = secondSeg.split(separator: ":").map(String.init)
        return poses + [seg.count == 2 ? ["\(seg[0])_shapes"] : ["\(secondSeg)_640"], [reid]]
    }

    public var description: String { name }
}

/// Whether the app is in the foreground and so may use the GPU: iOS forbids GPU work from the background without an
/// entitlement the app doesn't have. The app sets `foreground` from its scene phase (only `.active` counts); jobs read it
/// before every whole-frame pass (`JobOptions.gpuWide`). Thread-safe.
public final class ComputeGate: @unchecked Sendable {
    public static let shared = ComputeGate()
    private let lock = NSLock()
    private var _foreground: Bool

    public init(foreground: Bool = true) { _foreground = foreground }

    public var foreground: Bool {
        get { lock.withLock { _foreground } }
        set { lock.withLock { _foreground = newValue } }
    }
}

/// The loaded Core ML models of a ModelSet.
public final class PerceptionModels: @unchecked Sendable {
    /// Where the models run unless told otherwise: the Neural Engine (with the CPU for what it can't do), the setup the
    /// golden tests check against the lab; only the CPU in the Simulator, whose GPU breaks the re-ID net.
    public static var computeUnits: MLComputeUnits {
        #if targetEnvironment(simulator)
        .cpuOnly
        #else
        .cpuAndNeuralEngine
        #endif
    }

    /// Where each model's load is told (its start and how long it took), for an app's log: a load that never comes back
    /// on a phone then shows which model it was (build 114). Nil: not told.
    nonisolated(unsafe) public static var loadLog: (@Sendable (String) -> Void)?

    public let set: ModelSet
    let widePose: PoseNet
    let focusPose: PoseNet
    /// Optional: the app's download leaves the medium mask model out (masks are off, JobOptions.masks).
    let focusSeg: SegNet?
    /// The focus window's mask model is there (masks can be on).
    public var hasFocusMasks: Bool { focusSeg != nil }
    let secondPose: PoseNet
    let secondSeg: SegNet
    let reid: Embedder
    /// The whole-frame pose model's files (and function, for a multifunction model), for a second copy on the GPU
    /// (`gpuWidePose`).
    private let wideURLs: [SIMD2<Int>: (url: URL, function: String?)]
    /// The re-ID net for plain crops (`plainEmbedder`), loaded on first use.
    private let plainURL: URL?
    private let computeUnits: MLComputeUnits
    private var plain: Embedder?
    private var plainFailed = false
    private let gpuLock = NSLock()
    private var gpuWide: PoseNet?
    private var gpuWideFailed = false
    private var _gpuLoadSeconds = 0.0

    /// Compiled models from the app bundle (root or a `Models` folder).
    public convenience init(_ set: ModelSet = .default, bundle: Bundle = .main,
                            computeUnits: MLComputeUnits = PerceptionModels.computeUnits) throws {
        try self.init(set, computeUnits: computeUnits) { name in
            bundle.url(forResource: name, withExtension: "mlmodelc")
                ?? bundle.url(forResource: name, withExtension: "mlmodelc", subdirectory: "Models")
        }
    }

    /// Compiled models from `directory` (the downloaded full set), the rest from the app bundle (the small models
    /// second chance uses).
    public convenience init(_ set: ModelSet, directory: URL, bundle: Bundle,
                            computeUnits: MLComputeUnits = PerceptionModels.computeUnits) throws {
        try self.init(set, computeUnits: computeUnits) { name in
            let u = directory.appendingPathComponent(name + ".mlmodelc")
            if FileManager.default.fileExists(atPath: u.path) { return u }
            return bundle.url(forResource: name, withExtension: "mlmodelc")
                ?? bundle.url(forResource: name, withExtension: "mlmodelc", subdirectory: "Models")
        }
    }

    /// Compiled models (`.mlmodelc`) from a directory, e.g. $DIRECTE_DATA/models/coreml/app.
    public convenience init(_ set: ModelSet = .default, directory: URL,
                            computeUnits: MLComputeUnits = PerceptionModels.computeUnits) throws {
        try self.init(set, computeUnits: computeUnits) { name in
            let u = directory.appendingPathComponent(name + ".mlmodelc")
            return FileManager.default.fileExists(atPath: u.path) ? u : nil
        }
    }

    init(_ set: ModelSet, computeUnits: MLComputeUnits, locate: (String) -> URL?) throws {
        self.set = set
        self.computeUnits = computeUnits
        plainURL = locate(set.plainReid)
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        var cache: [String: MLModel] = [:]
        let place = computeUnits == .cpuOnly ? "CPU" : computeUnits == .cpuAndNeuralEngine ? "Neural Engine" : "units \(computeUnits.rawValue)"
        func open(_ url: URL, _ c: MLModelConfiguration, _ name: String) throws -> MLModel {
            let started = Date()
            Self.loadLog?("model \(name) (\(set.name), \(place)): loading")
            do {
                let m = try MLModel(contentsOf: url, configuration: c)
                Self.loadLog?(String(format: "model %@ (%@, %@): loaded in %.2f s", name, set.name, place, Date().timeIntervalSince(started)))
                return m
            } catch {
                throw PerceptionError.modelLoad(name, "\(error)")
            }
        }
        func load(_ name: String, required: Bool = true) throws -> MLModel? {
            if let m = cache[name] { return m }
            guard let url = locate(name) else {
                if required { throw PerceptionError.modelMissing(name) }
                return nil
            }
            let m = try open(url, config, name)
            cache[name] = m
            return m
        }
        func pose(_ stem: String, rect: Bool) throws -> PoseNet {
            if let url = locate("\(stem)_shapes") {
                var shapes: [SIMD2<Int>: MLModel] = [:]
                for shape in Self.shapes where rect || shape == SIMD2(640, 640) {
                    let key = "\(stem)_shapes/\(Self.function(shape))"
                    if cache[key] == nil {
                        let c = config.copy() as! MLModelConfiguration
                        c.functionName = Self.function(shape)
                        cache[key] = try open(url, c, key)
                    }
                    shapes[shape] = cache[key]
                }
                return PoseNet(models: shapes)
            }
            var shapes: [SIMD2<Int>: MLModel] = [SIMD2(640, 640): try load("\(stem)_640")!]
            if rect {
                if let m = try load("\(stem)_640x384", required: false) { shapes[SIMD2(640, 384)] = m }
                if let m = try load("\(stem)_384x640", required: false) { shapes[SIMD2(384, 640)] = m }
            }
            return PoseNet(models: shapes)
        }
        widePose = try pose(set.widePose, rect: true)
        var urls: [SIMD2<Int>: (url: URL, function: String?)] = [:]
        if let u = locate("\(set.widePose)_shapes") {
            for shape in Self.shapes { urls[shape] = (u, Self.function(shape)) }
        } else {
            for (shape, suffix) in [(SIMD2(640, 640), "640"), (SIMD2(640, 384), "640x384"), (SIMD2(384, 640), "384x640")] {
                if let u = locate("\(set.widePose)_\(suffix)") { urls[shape] = (u, nil) }
            }
        }
        wideURLs = urls
        focusPose = try pose(set.focusPose, rect: false)
        secondPose = try pose(set.secondPose, rect: false)
        // a mask model is `<stem>_640`, or `<stem>:<function>`: that function of the `<stem>_shapes` model (RTMO-s's box
        // finder shares its pose model's weights: one file in the bundle)
        func seg(_ name: String, required: Bool) throws -> SegNet? {
            let parts = name.split(separator: ":").map(String.init)
            guard parts.count == 2 else { return try load("\(name)_640", required: required).map { SegNet(model: $0) } }
            let key = "\(parts[0])_shapes/\(parts[1])"
            if cache[key] == nil {
                guard let url = locate("\(parts[0])_shapes") else {
                    if required { throw PerceptionError.modelMissing(name) }
                    return nil
                }
                let c = config.copy() as! MLModelConfiguration
                c.functionName = parts[1]
                cache[key] = try open(url, c, key)
            }
            return SegNet(model: cache[key]!)
        }
        focusSeg = try seg(set.focusSeg, required: false)
        secondSeg = try seg(set.secondSeg, required: true)!
        // The re-ID net gives all-zero embeddings on the iOS Simulator's GPU (and the tracker then never locks): if
        // it doesn't answer sensibly where it was put, run it on the CPU.
        var reidModel = try load(set.reid)!
        Self.loadLog?("model \(set.reid) (\(set.name), \(place)): checking its answers")
        if computeUnits != .cpuOnly, !Embedder.works(reidModel), let url = locate(set.reid) {
            let cpu = MLModelConfiguration()
            cpu.computeUnits = .cpuOnly
            reidModel = try MLModel(contentsOf: url, configuration: cpu)
        }
        reid = Embedder(model: reidModel)
        Self.loadLog?("models \(set.name) (\(place)): all loaded")
    }
}

extension PerceptionModels {
    /// The pose models' input shapes (H, W) and their functions in a `_shapes` multifunction model.
    static let shapes = [SIMD2(640, 640), SIMD2(640, 384), SIMD2(384, 640)]
    static func function(_ shape: SIMD2<Int>) -> String { "s\(shape.x)x\(shape.y)" }

    /// OSNet-IBN x1.0 trained on MSMT17 + Duke + CUHK03, the re-ID net for plain crops (no masks): it separates
    /// the climber better from plain crops than OSNet-AIN does with the 640 masks (macro AUC 0.951 vs 0.905,
    /// docs/review/deep-dives/reid-models.md), so the mask model can be left out (`JobOptions.masks`).
    public static let plainReID = "osnet_ibn_x1_0_msdc"

    /// The plain-crop re-ID net (`plainReID`), loaded on first use; nil if the app doesn't have it.
    func plainEmbedder() -> Embedder? {
        gpuLock.withLock {
            if let plain { return plain }
            guard !plainFailed, let plainURL else { return nil }
            let config = MLModelConfiguration()
            config.computeUnits = computeUnits
            do {
                var m = try MLModel(contentsOf: plainURL, configuration: config)
                if computeUnits != .cpuOnly, !Embedder.works(m) {  // the Simulator's GPU, as for the main re-ID net
                    let cpu = MLModelConfiguration()
                    cpu.computeUnits = .cpuOnly
                    m = try MLModel(contentsOf: plainURL, configuration: cpu)
                }
                plain = Embedder(model: m)
            } catch {
                plainFailed = true
            }
            return plain
        }
    }
    /// The whole-frame pose model loaded a second time for the GPU (`.cpuAndGPU`), loaded on first use (a second or so,
    /// more the first time on a phone). Nil in the Simulator, or if it didn't load (then it isn't tried again). Call it
    /// only in the foreground: loading prepares GPU work too.
    func gpuWidePose() -> PoseNet? {
        gpuLock.withLock {
            if let gpuWide { return gpuWide }
            if gpuWideFailed { return nil }
            #if targetEnvironment(simulator)
            gpuWideFailed = true
            return nil
            #else
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU
            let started = Date()
            do {
                var shapes: [SIMD2<Int>: MLModel] = [:]
                for (shape, src) in wideURLs {
                    let c = config.copy() as! MLModelConfiguration
                    c.functionName = src.function
                    shapes[shape] = try MLModel(contentsOf: src.url, configuration: c)
                }
                guard shapes[SIMD2(640, 640)] != nil else { throw PerceptionError.modelMissing("wide pose 640") }
                gpuWide = PoseNet(models: shapes)
            } catch {
                gpuWideFailed = true
            }
            _gpuLoadSeconds = Date().timeIntervalSince(started)
            return gpuWide
            #endif
        }
    }

    /// Loads the GPU copy of the whole-frame model now (at a job's start, in the foreground) rather than on its first
    /// pass; false if there's none.
    @discardableResult public func prepareGPUWide() -> Bool { gpuWidePose() != nil }

    /// How long loading the GPU copy took (0 if it wasn't loaded).
    public var gpuLoadSeconds: Double { gpuLock.withLock { _gpuLoadSeconds } }
}
