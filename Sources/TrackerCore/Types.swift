import Foundation

// Shared vocabulary of the engine. Perception produces `Candidate`s, TrackerCore turns them into
// `TrackPoint`s, Render turns those into a `CropPath` and a video. Mirrors the lab (Python):
// directe_tracking/tracker.py (Cands, Step), directe_tracking/appearance.py (constants).
// Coordinates are source-video pixels after applying the track's preferredTransform (the upright
// frame the user sees), origin top-left, y down.

/// Axis-aligned box: (x0, y0) top-left, (x1, y1) bottom-right.
public struct Box: Hashable, Codable, Sendable {
    public var x0: Double
    public var y0: Double
    public var x1: Double
    public var y1: Double

    public init(_ x0: Double, _ y0: Double, _ x1: Double, _ y1: Double) {
        self.x0 = x0; self.y0 = y0; self.x1 = x1; self.y1 = y1
    }

    public var width: Double { x1 - x0 }
    public var height: Double { y1 - y0 }
    public var area: Double { max(width, 0) * max(height, 0) }
    public var center: SIMD2<Double> { SIMD2((x0 + x1) / 2, (y0 + y1) / 2) }

    public func offset(_ d: SIMD2<Double>) -> Box { Box(x0 + d.x, y0 + d.y, x1 + d.x, y1 + d.y) }

    public func iou(_ o: Box) -> Double {
        let iw = max(0, min(x1, o.x1) - max(x0, o.x0)), ih = max(0, min(y1, o.y1) - max(y0, o.y0))
        let inter = iw * ih
        return inter / max(area + o.area - inter, 1e-6)
    }

    /// Probably the same person (lab `same_person`): overlap by IoU, or one box mostly inside the
    /// other and at least a quarter of its height (a detector that boxed only the upper body).
    public func samePerson(_ o: Box, iouThreshold: Double = 0.5) -> Bool {
        let iw = max(0, min(x1, o.x1) - max(x0, o.x0)), ih = max(0, min(y1, o.y1) - max(y0, o.y0))
        let contained = iw * ih / max(min(max(area, 1e-6), max(o.area, 1e-6)), 1e-6)
        let similar = min(height, o.height) >= 0.25 * max(height, o.height)
        return iou(o) >= iouThreshold || (contained >= 0.7 && similar)
    }
}

/// A COCO-17 keypoint (x, y in source pixels, conf in 0...1).
public struct Keypoint: Codable, Sendable {
    public var x: Float
    public var y: Float
    public var conf: Float
    public init(x: Float, y: Float, conf: Float) { self.x = x; self.y = y; self.conf = conf }
}

/// Colour-signature constants shared by Perception (extraction) and TrackerCore (comparison).
/// Must match directe_tracking/appearance.py VERSION 4.
public enum AppearanceSpec {
    public static let regions = ["head", "torso", "hips", "legs"]
    public static let weights: [Float] = [0.15, 0.35, 0.2, 0.3]
    public static let hueBins = 12
    public static let greyEdges: [Float] = [45, 90, 150, 200]  // V thresholds (0...255) for non-coloured pixels
    public static let dim = hueBins + greyEdges.count + 1      // 17
    public static let kpConf: Float = 0.35
    public static let minPixels = 15
    public static let unknown: Float = -1
    public static let minEvidence: Float = 0.3
    public static let vetoBelow: Float = 0.55
    public static let vetoGain: Float = 0.8
}

/// One person in one frame, described for the tracker (lab: one row of `Cands`).
public struct Candidate: Sendable {
    public var box: Box
    /// Colour histograms per region, `AppearanceSpec.regions.count` x `AppearanceSpec.dim`.
    public var feats: [[Float]]
    /// Which regions could be placed (from keypoints).
    public var valids: [Bool]
    /// L2-normalised re-ID embedding, if the re-ID model ran.
    public var emb: [Float]?
    /// COCO-17 keypoints; empty for people only the mask model found.
    public var keypoints: [Keypoint]

    public init(box: Box, feats: [[Float]], valids: [Bool], emb: [Float]?, keypoints: [Keypoint] = []) {
        self.box = box; self.feats = feats; self.valids = valids; self.emb = emb; self.keypoints = keypoints
    }
}

public enum TrackState: String, Codable, Sendable {
    case locked, uncertain, lost
}

/// What the tracker says at one tick (time `t` in seconds of the source video).
/// The climber's wrists relative to their hip centre, in body heights (their box height), each nil when not seen;
/// and their ankles in image pixels (where they stand on the wall: subtract the camera offset).
public struct Wrists: Codable, Sendable, Equatable {
    public var left: SIMD2<Double>?
    public var right: SIMD2<Double>?
    public var leftAnkle: SIMD2<Double>?
    public var rightAnkle: SIMD2<Double>?

    public init(left: SIMD2<Double>?, right: SIMD2<Double>?, leftAnkle: SIMD2<Double>? = nil, rightAnkle: SIMD2<Double>? = nil) {
        self.left = left; self.right = right; self.leftAnkle = leftAnkle; self.rightAnkle = rightAnkle
    }
}

public struct TrackPoint: Codable, Sendable {
    public var t: Double
    public var state: TrackState
    public var box: Box
    public init(t: Double, state: TrackState, box: Box) { self.t = t; self.state = state; self.box = box }
}

/// The crop window (source pixels) for each output frame time.
public struct CropPath: Codable, Sendable {
    public var times: [Double]
    public var windows: [Box]
    public init(times: [Double], windows: [Box]) { self.times = times; self.windows = windows }
}
