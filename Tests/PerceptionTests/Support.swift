import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import ImageIO
@testable import Perception

// Fixture and frame loading for the Perception tests. Fixtures are read by path (no package
// resources): the unit fixtures here, the golden ones (recorded on real footage) from $DIRECTE_GOLDEN. Frames and
// compiled models come from $DIRECTE_DATA (ios_golden/perception, models/coreml/app); the tests that need them are
// skipped without it.

enum Golden {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
    static let golden = ProcessInfo.processInfo.environment["DIRECTE_GOLDEN"].map { URL(fileURLWithPath: $0) }
    static let data = URL(fileURLWithPath: ProcessInfo.processInfo.environment["DIRECTE_DATA"] ?? "/nonexistent")
    static let frames = data.appendingPathComponent("ios_golden/perception")
    static let models = data.appendingPathComponent("models/coreml/app")

    /// Where the models run: DIRECTE_UNITS=all|cpu|gpu (xcodebuild passes TEST_RUNNER_DIRECTE_UNITS); default
    /// CPU + Neural Engine, as the app runs on a phone.
    static var units: MLComputeUnits {
        switch ProcessInfo.processInfo.environment["DIRECTE_UNITS"] {
        case "all": .all
        case "cpu": .cpuOnly
        case "gpu": .cpuAndGPU
        default: .cpuAndNeuralEngine
        }
    }

    static var haveFrames: Bool {
        FileManager.default.fileExists(atPath: frames.path) && golden.map { FileManager.default.fileExists(atPath: $0.path) } == true
    }
    static var haveModels: Bool { FileManager.default.fileExists(atPath: models.appendingPathComponent("yolo26n-pose_640.mlmodelc").path) }

    static func json(_ name: String) throws -> [String: Any] {
        let here = fixtures.appendingPathComponent(name)
        let url = FileManager.default.fileExists(atPath: here.path) ? here : (golden ?? fixtures).appendingPathComponent(name)
        let d = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: d) as! [String: Any]
    }

    static func bytes(_ b64: Any?) -> [UInt8] { [UInt8](Data(base64Encoded: b64 as! String)!) }

    /// A PNG (as cv2.imwrite wrote it) as a 32BGRA pixel buffer, without colour management.
    static func pixelBuffer(png url: URL) throws -> CVPixelBuffer {
        let src = CGImageSourceCreateWithURL(url as CFURL, nil)!
        let img = CGImageSourceCreateImageAtIndex(src, 0, nil)!
        let w = img.width, h = img.height
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        CVPixelBufferLockBaseAddress(pb!, [])
        let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb!), width: w, height: h, bitsPerComponent: 8,
                            bytesPerRow: CVPixelBufferGetBytesPerRow(pb!), space: img.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        CVPixelBufferUnlockBaseAddress(pb!, [])
        return pb!
    }

    /// A BGR byte array (cv2 layout) as an owned BGRA image.
    static func image(bgr: [UInt8], width: Int, height: Int) -> BGRAImage {
        let img = BGRAImage(width: width, height: height)
        for i in 0..<(width * height) {
            img.base[4 * i] = bgr[3 * i]; img.base[4 * i + 1] = bgr[3 * i + 1]; img.base[4 * i + 2] = bgr[3 * i + 2]
            img.base[4 * i + 3] = 255
        }
        return img
    }
}

func f32(_ a: Any?) -> Float { Float((a as! NSNumber).doubleValue) }
func floats(_ a: Any?) -> [Float] { (a as! [Any]).map { f32($0) } }
func box4(_ a: Any?) -> SIMD4<Float> { let v = floats(a); return SIMD4(v[0], v[1], v[2], v[3]) }
func keypoints(_ a: Any?) -> [Keypoint] {
    (a as! [Any]).map { let v = floats($0); return Keypoint(x: v[0], y: v[1], conf: v[2]) }
}
func polygon(_ a: Any?) -> [SIMD2<Double>]? {
    guard let pts = a as? [Any] else { return nil }
    return pts.map { let v = ($0 as! [Any]).map { ($0 as! NSNumber).doubleValue }; return SIMD2(v[0], v[1]) }
}

func bhattacharyya(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).reduce(0) { $0 + ($1.0 * $1.1).squareRoot() } }
func cosine(_ a: [Float], _ b: [Float]) -> Float {
    let d = zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
    let na = a.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot(), nb = b.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
    return d / max(na * nb, 1e-9)
}

/// Greedy one-to-one matching by IoU (best pairs first); returns (i in a, j in b, iou).
func matchBoxes(_ a: [Box], _ b: [Box], minIoU: Double) -> [(Int, Int, Double)] {
    var pairs: [(Int, Int, Double)] = []
    for i in a.indices { for j in b.indices { let o = a[i].iou(b[j]); if o >= minIoU { pairs.append((i, j, o)) } } }
    pairs.sort { $0.2 > $1.2 }
    var ua = Set<Int>(), ub = Set<Int>(), out: [(Int, Int, Double)] = []
    for p in pairs where !ua.contains(p.0) && !ub.contains(p.1) { out.append(p); ua.insert(p.0); ub.insert(p.1) }
    return out
}
