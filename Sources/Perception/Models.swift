import CoreML
import Foundation

/// Which Core ML models run in each pass. Names are weight stems; the files are compiled models
/// named `<stem>_640.mlmodelc` (640x640 input), `<stem>_640x384.mlmodelc` / `<stem>_384x640.mlmodelc`
/// (H x W, the wide pass on 16:9 portrait / landscape frames, optional) and `<reid>.mlmodelc`
/// (tools/export_coreml.py). A pose model's shapes can instead be one multifunction model,
/// `<stem>_shapes.mlmodelc` (functions `s<H>x<W>`, the weights stored once), which is preferred.
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

    public init(name: String, widePose: String, focusPose: String, focusSeg: String,
                secondPose: String, secondSeg: String, reid: String) {
        self.name = name; self.widePose = widePose; self.focusPose = focusPose; self.focusSeg = focusSeg
        self.secondPose = secondPose; self.secondSeg = secondSeg; self.reid = reid
    }

    /// lab pipeline "lite": yolo26n pose + seg at 640, OSNet x0.25.
    public static let lite = ModelSet(name: "lite", widePose: "yolo26n-pose", focusPose: "yolo26n-pose",
                                      focusSeg: "yolo26n-seg", secondPose: "yolo26n-pose", secondSeg: "yolo26n-seg",
                                      reid: "osnet_x0_25_msmt17")
    /// The app default (docs/review/deep-dives/detector-size.md): yolo26m pose + seg at 640 in the
    /// wide and focus passes, the nano models for second chance (same recall at 2x, ~20 ms cheaper),
    /// OSNet-AIN re-ID. (The lab's "mid" pipeline also uses yolo26m for second chance.)
    public static let mid = ModelSet(name: "mid", widePose: "yolo26m-pose", focusPose: "yolo26m-pose",
                                     focusSeg: "yolo26m-seg", secondPose: "yolo26n-pose", secondSeg: "yolo26n-seg",
                                     reid: "osnet_ain_x1_0_msmt17")
    public static let `default` = mid

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
        plainURL = locate(Self.plainReID)
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        var cache: [String: MLModel] = [:]
        func load(_ name: String, required: Bool = true) throws -> MLModel? {
            if let m = cache[name] { return m }
            guard let url = locate(name) else {
                if required { throw PerceptionError.modelMissing(name) }
                return nil
            }
            let m = try MLModel(contentsOf: url, configuration: config)
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
                        cache[key] = try MLModel(contentsOf: url, configuration: c)
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
        focusSeg = try load("\(set.focusSeg)_640", required: false).map { SegNet(model: $0) }
        secondSeg = SegNet(model: try load("\(set.secondSeg)_640")!)
        // The re-ID net gives all-zero embeddings on the iOS Simulator's GPU (and the tracker then never locks): if
        // it doesn't answer sensibly where it was put, run it on the CPU.
        var reidModel = try load(set.reid)!
        if computeUnits != .cpuOnly, !Embedder.works(reidModel), let url = locate(set.reid) {
            let cpu = MLModelConfiguration()
            cpu.computeUnits = .cpuOnly
            reidModel = try MLModel(contentsOf: url, configuration: cpu)
        }
        reid = Embedder(model: reidModel)
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
