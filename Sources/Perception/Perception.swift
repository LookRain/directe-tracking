@_exported import TrackerCore

// Core ML person detection (pose + masks), re-ID embeddings, colour features and camera motion:
// the Swift side of directe_tracking/online.py `perceive()` / `second_chance()` / CameraMotion, directe_tracking/identity.py
// and directe_tracking/appearance.py `features()`. See ../../README.md for the public API.
//
//   let models = try PerceptionModels(.mid)            // compiled models from the app bundle
//   let perceiver = Perceiver(models: models)
//   let cands = try perceiver.perceive(frame, focus: Perceiver.focusWindow(around: pb, width: W, height: H))
//
// Frames are 32BGRA CVPixelBuffers in upright source pixels (the lab's BGR frames).
public enum Perception {}

/// Something that can say whether the wall moved at a box (directe_tracking/motion.py Evidence.supports), used
/// by `Perceiver.secondChance` to reject static boxes the 2x view hallucinates on holds and rock.
public protocol MotionEvidence: AnyObject {
    func supports(_ box: Box) -> Bool
}
