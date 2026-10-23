import CoreVideo
import Foundation
import Testing
@testable import Perception

// directe_tracking/motion.py building blocks against perception_units.json, plus Vision registration on a known shift.

@Suite struct MotionPorts {
    let u: [String: Any]
    init() throws { u = try Golden.json("perception_units.json") }

    /// The procedural difference image tools/ios_golden/perception_golden.py builds.
    static func diffImage() -> [Float] {
        let W = WallMotion.width, H = WallMotion.height
        var d = [Float](repeating: 0, count: W * H)
        for y in 0..<H {
            for x in 0..<W {
                let fx = Float(x), fy = Float(y)
                var v: Float = 0.05 + 0.04 * sin(fx * 0.37) * cos(fy * 0.23)
                for (cx, cy, r, a) in [(60, 100, 9, 1.2), (75, 118, 6, 0.9), (200, 300, 14, 1.5), (140, 420, 3, 2.0), (30, 400, 25, 1.0)] as [(Float, Float, Float, Float)] {
                    v += a * exp(-((fx - cx) * (fx - cx) + (fy - cy) * (fy - cy)) / (2 * r * r))
                }
                d[y * W + x] = v
            }
        }
        return d
    }

    @Test func blurAndBlobsMatchLab() {
        let mb = u["motion_blobs"] as! [String: Any]
        let W = WallMotion.width, H = WallMotion.height
        let db = gaussianBlur(Self.diffImage(), W, H, sigma: 1)
        let xy = mb["xy"] as! [[Int]], want = floats(mb["blur_samples"])
        let err = zip(xy, want).map { abs(db[$0.0[1] * W + $0.0[0]] - $0.1) }.max()!
        #expect(err < 1e-4, "GaussianBlur sigma 1: max error \(err)")
        for case let c as [String: Any] in mb["cases"] as! [Any] {
            let h = c["h"] as! Double
            let got = WallMotion.blobs(db, W, H, hExp: h)
            let lab = (c["boxes"] as! [[Double]]).map { Box($0[0], $0[1], $0[2], $0[3]) }
            #expect(got == lab, "blobs(h=\(h)): \(got) vs lab \(lab)")
        }
    }

    @Test(.enabled(if: Golden.haveFrames)) func normaliseMatchesLab() throws {
        let mn = u["motion_norm"] as! [String: Any]
        let pb = try Golden.pixelBuffer(png: Golden.frames.appendingPathComponent(mn["frame"] as! String))
        let small = BGRAImage(width: WallMotion.width, height: WallMotion.height)
        try withPixels(pb) { cvResize($0, small.view, .area) }
        var sum = 0
        for i in 0..<(small.width * small.height) { for c in 0..<3 { sum += Int(small.base[4 * i + c]) } }
        #expect(sum == mn["small_sum"] as! Int, "INTER_AREA 4x: pixel sum \(sum) vs lab")
        let n = WallMotion.normalise(small.view, sigma: 25)
        var worst: Float = 0
        for (p, v) in zip(mn["xy"] as! [[Int]], mn["n"] as! [Any]) {
            let lab = floats(v)
            for c in 0..<3 { worst = max(worst, abs(n[3 * (p[1] * small.width + p[0]) + c] - lab[c])) }
        }
        #expect(worst < 1e-3, "norm: max error \(worst)")
    }

    /// Moving camera: the registered difference of a frame with itself shifted is ~0 except at the
    /// edges, and a pasted-in "climber" shows up as a blob.
    @Test(.enabled(if: Golden.haveFrames)) func registeredDifferenceFindsTheMover() throws {
        let base = try Golden.pixelBuffer(png: Golden.frames.appendingPathComponent("IMG_4826_003000.png"))
        func shifted(_ dx: Int, _ dy: Int, paste: Bool) throws -> CVPixelBuffer {
            var out: CVPixelBuffer?
            CVPixelBufferCreate(nil, 1080, 1920, kCVPixelFormatType_32BGRA, nil, &out)
            try withPixels(base) { src in
                try withPixels(out!, readOnly: false) { dst in
                    for y in 0..<1920 {
                        for x in 0..<1080 {
                            let sx = min(max(x - dx, 0), 1079), sy = min(max(y - dy, 0), 1919)
                            for c in 0..<4 { dst.row(y)[4 * x + c] = src.row(sy)[4 * sx + c] }
                            if paste && x >= 500 && x < 620 && y >= 900 && y < 1200 { dst.row(y)[4 * x] = 255; dst.row(y)[4 * x + 1] = 40; dst.row(y)[4 * x + 2] = 200 }
                        }
                    }
                }
            }
            return out!
        }
        let m = WallMotion(movingCamera: true)
        let pred = Box(480, 880, 640, 1220)
        try m.update(t: 0, frame: try shifted(0, 0, paste: false), predicted: pred)
        try m.update(t: 1.0, frame: try shifted(24, -16, paste: true), predicted: pred)
        print("moving-camera blobs: \(m.blobs)")
        #expect(m.supports(Box(500, 900, 620, 1200)))
        #expect(!m.supports(Box(100, 200, 220, 500)))
    }
}
